import os
import json
import logging
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from transformers import BertTokenizer
from utils.options import Args
from utils.dataset_utils import NERDataset
from utils.model_utils import build_model
from utils.evaluator import mrc_evaluation
from preprocess.processor import cut_sent, fine_grade_tokenize
from preprocess.processor import NERProcessor, convert_examples_to_features

MID_DATA_DIR = "./data/ChiFinAnn/fold5/MRC/mid_data"
RAW_DATA_DIR = "./data/ChiFinAnn/fold5/MRC/raw_data"
GPU_IDS = "0"
MAX_SEQ_LEN = 512
BERT_DIR = "./FinBERT/"
MODEL_PATH = "./out/finbert_wd_fp16_ls_ce/checkpoint-4540/model.pt"
TEST_SAVE_PATH = "./out/finbert_wd_fp16_ls_ce_mrc/test_metric.txt"

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def predict(opt):
    with open(os.path.join(opt.mid_data_dir, f'ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)

    model = build_model(opt.bert_dir,
                        dropout_prob=opt.dropout_prob,
                        loss_type=opt.loss_type)

    processor = NERProcessor(opt.max_seq_len - max(list(map(len, ent2id.values()))) - 3)

    test_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'test.json'))
    test_examples = processor.get_examples(test_raw_examples, 'test')
    test_features, test_callback_info = convert_examples_to_features(test_examples, opt.max_seq_len,
                                                                     opt.bert_dir, ent2id)
    test_dataset = NERDataset(test_features, 'test', use_type_embed=opt.use_type_embed)
    test_loader = DataLoader(test_dataset, batch_size=opt.eval_batch_size,
                             shuffle=False, num_workers=1, pin_memory=True)
    test_info = (test_loader, test_callback_info)

    device = torch.device("cpu" if opt.gpu_ids.split(',')[0] == '-1' else "cuda:" + opt.gpu_ids.split(',')[0])
    model.load_state_dict(torch.load(opt.model_path, map_location=torch.device('cpu')), strict=False)
    model.to(device)

    metric_str, f1 = mrc_evaluation(model, test_info, device)

    logger.info(f'The test F1 score:{f1} \n')
    logger.info(f'Metric info:{metric_str}')

    with open(opt.test_save_path, 'a', encoding='utf-8') as f1:
        f1.write(metric_str)


if __name__ == '__main__':
    logger.info("------Start Prediction------")
    args = Args().get_parser()

    args.mid_data_dir = MID_DATA_DIR
    args.raw_data_dir = RAW_DATA_DIR
    args.gpu_ids = GPU_IDS
    args.model_path = MODEL_PATH
    args.bert_dir = BERT_DIR
    args.test_save_path = TEST_SAVE_PATH

    predict(args)
