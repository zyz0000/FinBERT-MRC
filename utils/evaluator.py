import torch
import logging
import numpy as np
from collections import defaultdict
from preprocess.processor import ENTITY_TYPES

logger = logging.getLogger(__name__)


def get_base_out(model, loader, device):
    """
    每一个任务的 forward 都一样，封装起来
    """
    model.eval()
    with torch.no_grad():
        for idx, _batch in enumerate(loader):
            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)
            tmp_out = model(**_batch)
            yield tmp_out


# 严格解码 baseline
def mrc_decode(start_logits, end_logits, raw_text):
    predict_entities = []
    start_pred = np.argmax(start_logits, -1)
    end_pred = np.argmax(end_logits, -1)

    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i:i+j+1]
                predict_entities.append((tmp_ent, i))
                break

    return predict_entities


def calculate_metric(gt, predict):
    """
    计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp

    return np.array([tp, fp, fn])


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def mrc_evaluation(model, dev_info, device):
    dev_loader, (dev_callback_info, type_weight) = dev_info
    start_logits, end_logits = None, None

    model.eval()
    for tmp_pred in get_base_out(model, dev_loader, device):
        tmp_start_logits = tmp_pred[0].cpu().numpy()
        tmp_end_logits = tmp_pred[1].cpu().numpy()

        if start_logits is None:
            start_logits = tmp_start_logits
            end_logits = tmp_end_logits
        else:
            start_logits = np.append(start_logits, tmp_start_logits, axis=0)
            end_logits = np.append(end_logits, tmp_end_logits, axis=0)

    assert len(start_logits) == len(end_logits) == len(dev_callback_info)

    role_metric = np.zeros([len(ENTITY_TYPES), 3])
    mirco_metrics = np.zeros(3)

    id2ent = {x: i for i, x in enumerate(ENTITY_TYPES)}

    for tmp_start_logits, tmp_end_logits, tmp_callback \
            in zip(start_logits, end_logits, dev_callback_info):
        text, text_offset, ent_type, gt_entities = tmp_callback
        tmp_start_logits = tmp_start_logits[text_offset:text_offset+len(text)]
        tmp_end_logits = tmp_end_logits[text_offset:text_offset+len(text)]
        pred_entities = mrc_decode(tmp_start_logits, tmp_end_logits, text)
        role_metric[id2ent[ent_type]] += calculate_metric(gt_entities, pred_entities)

    for idx, _type in enumerate(ENTITY_TYPES):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])
        mirco_metrics += temp_metric * type_weight[_type]

    metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                  f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]
