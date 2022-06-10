import time
import os
import json
import logging
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from sklearn.model_selection import KFold
from utils.trainer import train
from utils.options import Args
from utils.model_utils import build_model
from utils.dataset_utils import NERDataset
from utils.evaluator import mrc_evaluation
from utils.functions_utils import set_seed, get_model_path_list, load_model_and_parallel, get_time_dif
from preprocess.processor import NERProcessor, convert_examples_to_features

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def training(opt):
    with open(os.path.join(opt.mid_data_dir, f'ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)

    processor = NERProcessor(opt.max_seq_len - max(list(map(len, ent2id.values()))) - 3)

    train_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'train.json'))
    train_examples = processor.get_examples(train_raw_examples, 'train')
    train_features = convert_examples_to_features(train_examples, opt.max_seq_len,
                                                  opt.bert_dir, ent2id)[0]
    train_dataset = NERDataset(train_features, 'train', use_type_embed=opt.use_type_embed)

    model = build_model(opt.bert_dir,
                        dropout_prob=opt.dropout_prob,
                        loss_type=opt.loss_type)

    train(opt, model, train_dataset)


def validate(opt):
    with open(os.path.join(opt.mid_data_dir, f'ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)

    model = build_model(opt.bert_dir,
                        dropout_prob=opt.dropout_prob,
                        loss_type=opt.loss_type)

    processor = NERProcessor(opt.max_seq_len - max(list(map(len, ent2id.values()))) - 3)

    dev_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'dev.json'))
    dev_examples = processor.get_examples(dev_raw_examples, 'dev')
    dev_features, dev_callback_info = convert_examples_to_features(dev_examples, opt.max_seq_len,
                                                                   opt.bert_dir, ent2id)
    dev_dataset = NERDataset(dev_features, 'dev', use_type_embed=opt.use_type_embed)
    dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    dev_info = (dev_loader, dev_callback_info)
    
    device = torch.device("cpu" if opt.gpu_ids.split(',')[0] == '-1' else "cuda:" + opt.gpu_ids.split(',')[0])
    model_path_list = get_model_path_list(opt.output_dir)

    metric_str = ''
    max_f1 = 0.
    max_f1_step = 0
    max_f1_path = ''

    for idx, model_path in enumerate(model_path_list):
        tmp_step = model_path.split('/')[-2].split('-')[-1]
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        model.to(device)

        tmp_metric_str, tmp_f1 = mrc_evaluation(model, dev_info, device)

        logger.info(f'In step {tmp_step}:\n {tmp_metric_str}')
        metric_str += f'In step {tmp_step}:\n {tmp_metric_str}' + '\n\n'

        if tmp_f1 > max_f1:
            max_f1 = tmp_f1
            max_f1_step = tmp_step
            max_f1_path = model_path

    max_metric_str = f'Max f1 is: {max_f1}, in step {max_f1_step}'
    logger.info(max_metric_str)
    metric_str += max_metric_str + '\n'
    eval_save_path = os.path.join(opt.output_dir, 'eval_metric.txt')

    with open(eval_save_path, 'a', encoding='utf-8') as f1:
        f1.write(metric_str)

    with open('./best_ckpt_path.txt', 'a', encoding='utf-8') as f2:
        f2.write(max_f1_path + '\n')

    del_dir_list = [os.path.join(opt.output_dir, path.split('/')[-2])
                    for path in model_path_list if path != max_f1_path]

    #return del_dir_list
    import shutil
    for x in del_dir_list:
        shutil.rmtree(x)
        logger.info('{}已删除'.format(x))


if __name__ == '__main__':
    start_time = time.time()
    args = Args().get_parser()

    logging.info('----------------开始计时----------------')
    logging.info('----------------------------------------')

    args.output_dir = os.path.join(args.output_dir, args.bert_type)

    if args.attack_train != '':
        args.output_dir += f'_{args.attack_train}'

    if args.weight_decay:
        args.output_dir += '_wd'

    if args.use_fp16:
        args.output_dir += '_fp16'

    args.output_dir += f'_{args.loss_type}'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f'MRC in max_seq_len {args.max_seq_len}')

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    set_seed(args.seed)
    training(args)
    dist.barrier()
    if args.local_rank == 0:
        validate(args)

    time_dif = get_time_dif(start_time)
    logging.info("----------本次容器运行时长：{}-----------".format(time_dif))

    dist.destroy_process_group()
