import os
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from datetime import timedelta
import time
import logging

logger = logging.getLogger(__name__)


def get_time_dif(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def set_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def load_model_and_parallel(opt, model, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """
    gpu_ids = opt.gpu_ids.split(',')

    # set to device to the first cuda
    if gpu_ids[0] == '-1':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", opt.local_rank)
    gpu_ids = list(map(int, gpu_ids))

    if ckpt_path is not None:
        logger.info(f'Load ckpt from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)
    model.to(device)

    if len(gpu_ids) > 1:
        logger.info(f'Use multi gpus in: {gpu_ids}')
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[opt.local_rank],
                                                          output_device=opt.local_rank,
                                                          find_unused_parameters=True)
    else:
        logger.info(f'Use single gpu in: {gpu_ids}')
    logger.info("Model loaded successfully!")

    return model, device


def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_lists = []

    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            if 'model.pt' == _file:
                model_lists.append(os.path.join(root, _file))

    model_lists = sorted(model_lists,
                         key=lambda x: (x.split('/')[-3], int(x.split('/')[-2].split('-')[-1])))

    return model_lists


def swa(model, model_dir, swa_start=1):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)

    assert 1 <= swa_start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:]:
            logger.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())
            alpha = 1. / (swa_n + 1.)
            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))
            swa_n += 1

    # use 100000 to represent swa to avoid clash
    swa_model_dir = os.path.join(model_dir, f'checkpoint-100000')
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)
    logger.info(f'Save swa model in: {swa_model_dir}')
    swa_model_path = os.path.join(swa_model_dir, 'model.pt')
    if torch.__version__ == "1.10.1+cu102":
        torch.save(swa_model.state_dict(), swa_model_path)
    else:
        torch.save(swa_model.state_dict(),
                   swa_model_path,
                   _use_new_zipfile_serialization=False)

    return swa_model

