import os
import re
import json
import logging
from transformers import BertTokenizer
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

ENTITY_TYPES = [
  "Price",
  "Shares",
  "Institution",
  "Company",
  "StockAbbr",
  "StockCode",
  "Date",
  "Ratio",
  "EquityHolder",
  "Pledgee"
]

"""
ENTITY_TYPES = [
  "mid1",
  "mid4",
  "mid5",
  "mid15",
  "mid16",
  "mid16_1",
  "mid17",
]
"""


class InputExample:
    def __init__(self,
                 set_type,
                 text,
                 labels=None,
                 distant_labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels
        self.distant_labels = distant_labels


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class MRCFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 ent_type=None,
                 start_ids=None,
                 end_ids=None):
        super(MRCFeature, self).__init__(token_ids=token_ids,
                                         attention_masks=attention_masks,
                                         token_type_ids=token_type_ids)
        self.ent_type = ent_type
        self.start_ids = start_ids
        self.end_ids = end_ids


class NERProcessor:
    def __init__(self, cut_sent_len=256):
        self.cut_sent_len = cut_sent_len

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = json.load(f)
        return raw_examples

    @staticmethod
    def _refactor_labels(sent, labels, distant_labels, start_index):
        """
        分句后需要重构 labels 的 offset
        :param sent: 切分并重新合并后的句子
        :param labels: 原始文档级的 labels
        :param distant_labels: 远程监督 label
        :param start_index: 该句子在文档中的起始 offset
        :return (type, entity, offset)
        """
        new_labels, new_distant_labels = [], []
        end_index = start_index + len(sent)

        for _label in labels:
            if start_index <= _label[2] <= _label[3] <= end_index:
                new_offset = _label[2] - start_index
                assert sent[new_offset: new_offset + len(_label[-1])] == _label[-1]

                new_labels.append((_label[1], _label[-1], new_offset))
            # label 被截断的情况
            elif _label[2] < end_index < _label[3]:
                raise RuntimeError(f'{sent}, {_label}')

        for _label in distant_labels:
            if _label in sent:
                new_distant_labels.append(_label)

        return new_labels, new_distant_labels

    def get_examples(self, raw_examples, set_type):
        examples = []

        for i, item in enumerate(raw_examples):
            text = item['text']
            distant_labels = item['candidate_entities']
            sentences = cut_sent(text, self.cut_sent_len)
            start_index = 0

            for sent in sentences:
                try:
                    labels, tmp_distant_labels = self._refactor_labels(sent, item['labels'], distant_labels, start_index)
                except AssertionError:
                    continue

                start_index += len(sent)

                examples.append(InputExample(set_type=set_type,
                                             text=sent,
                                             labels=labels,
                                             distant_labels=tmp_distant_labels))

        return examples


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens


def cut_sentences_v1(sent):
    """
    the first rank of sentence cut
    """
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    return sent.split("\n")


def cut_sentences_v2(sent):
    """
    the second rank of spilt sentence, split '；' | ';'
    """
    sent = re.sub('([；;])([^”’])', r"\1\n\2", sent)
    return sent.split("\n")


def cut_sent(text, max_seq_len):
    # 将句子分句，细粒度分句后再重新合并
    sentences = []

    # 细粒度划分
    sentences_v1 = cut_sentences_v1(text)
    for sent_v1 in sentences_v1:
        if len(sent_v1) > max_seq_len - 2:
            sentences_v2 = cut_sentences_v2(sent_v1)
            sentences.extend(sentences_v2)
        else:
            sentences.append(sent_v1)

    assert ''.join(sentences) == text

    # 合并
    merged_sentences = []
    start_index_ = 0

    while start_index_ < len(sentences):
        tmp_text = sentences[start_index_]
        end_index_ = start_index_ + 1
        while end_index_ < len(sentences) and \
                len(tmp_text) + len(sentences[end_index_]) <= max_seq_len - 2:
            tmp_text += sentences[end_index_]
            end_index_ += 1
        start_index_ = end_index_
        merged_sentences.append(tmp_text)

    return merged_sentences


def sent_mask(sent, stop_mask_range_list, mask_prob=0.15):
    """
    将句子中的词以 mask prob 的概率随机 mask，
    其中  85% 概率被置为 [mask] 15% 的概率不变。
    :param sent: list of segment words
    :param stop_mask_range_list: 不能 mask 的区域
    :param mask_prob: max mask nums: len(sent) * max_mask_prob
    :return:
    """
    max_mask_token_nums = int(len(sent) * mask_prob)
    mask_nums = 0
    mask_sent = []

    for i in range(len(sent)):
        flag = False
        for _stop_range in stop_mask_range_list:
            if _stop_range[0] <= i <= _stop_range[1]:
                flag = True
                break

        if flag:
            mask_sent.append(sent[i])
            continue

        if mask_nums < max_mask_token_nums:
            # mask_prob 的概率进行 mask, 80% 概率被置为 [mask]，10% 概率被替换， 10% 的概率不变
            if random.random() < mask_prob:
                mask_sent.append('[MASK]')
                mask_nums += 1
            else:
                mask_sent.append(sent[i])
        else:
            mask_sent.append(sent[i])

    return mask_sent


def convert_mrc_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                        max_seq_len, ent2id, ent2query, mask_prob=None):
    set_type = example.set_type
    text_b = example.text
    entities = example.labels

    features = []
    callback_info = []

    tokens_b = fine_grade_tokenize(text_b, tokenizer)
    assert len(tokens_b) == len(text_b)

    label_dict = defaultdict(list)

    for ent in entities:
        ent_type = ent[0]
        ent_start = ent[-1]
        ent_end = ent_start + len(ent[1]) - 1
        label_dict[ent_type].append((ent_start, ent_end, ent[1]))

    # 训练数据中构造
    if set_type == 'train':
        # 每一类为一个 example
        # for _type in label_dict.keys():
        for _type in ENTITY_TYPES:
            start_ids = [0] * len(tokens_b)
            end_ids = [0] * len(tokens_b)
            stop_mask_ranges = []
            text_a = ent2query[_type]
            tokens_a = fine_grade_tokenize(text_a, tokenizer)

            for _label in label_dict[_type]:
                start_ids[_label[0]] = 1
                end_ids[_label[1]] = 1
                stop_mask_ranges.append((_label[0], _label[1]))

            if len(start_ids) > max_seq_len - len(tokens_a) - 3:
                start_ids = start_ids[:max_seq_len - len(tokens_a) - 3]
                end_ids = end_ids[:max_seq_len - len(tokens_a) - 3]
                print('产生了不该有的截断')

            start_ids = [0] + [0] * len(tokens_a) + [0] + start_ids + [0]
            end_ids = [0] + [0] * len(tokens_a) + [0] + end_ids + [0]

            # pad
            if len(start_ids) < max_seq_len:
                pad_length = max_seq_len - len(start_ids)
                start_ids = start_ids + [0] * pad_length  # CLS SEP PAD label都为O
                end_ids = end_ids + [0] * pad_length

            assert len(start_ids) == max_seq_len
            assert len(end_ids) == max_seq_len

            # 随机mask
            if mask_prob:
                tokens_b = sent_mask(tokens_b, stop_mask_ranges, mask_prob=mask_prob)

            encode_dict = tokenizer.encode_plus(text=tokens_a,
                                                text_pair=tokens_b,
                                                max_length=max_seq_len,
                                                pad_to_max_length=True,
                                                truncation_strategy='only_second',
                                                is_pretokenized=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True,
                                                truncation=True)

            token_ids = encode_dict['input_ids']
            attention_masks = encode_dict['attention_mask']
            token_type_ids = encode_dict['token_type_ids']

            feature = MRCFeature(token_ids=token_ids,
                                 attention_masks=attention_masks,
                                 token_type_ids=token_type_ids,
                                 ent_type=ent2id[_type],
                                 start_ids=start_ids,
                                 end_ids=end_ids)
            features.append(feature)

    # 测试数据构造，为每一类单独构造一个 example
    else:
        for _type in ENTITY_TYPES:
            text_a = ent2query[_type]
            tokens_a = fine_grade_tokenize(text_a, tokenizer)

            encode_dict = tokenizer.encode_plus(text=tokens_a,
                                                text_pair=tokens_b,
                                                max_length=max_seq_len,
                                                pad_to_max_length=True,
                                                truncation_strategy='only_second',
                                                is_pretokenized=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True,
                                                truncation=True)

            token_ids = encode_dict['input_ids']
            attention_masks = encode_dict['attention_mask']
            token_type_ids = encode_dict['token_type_ids']

            tmp_callback = (text_b, len(tokens_a) + 2, _type)  # (text, text_offset, type, labels)
            tmp_callback_labels = []
            for _label in label_dict[_type]:
                tmp_callback_labels.append((_label[2], _label[0]))
            tmp_callback += (tmp_callback_labels, )
            callback_info.append(tmp_callback)

            feature = MRCFeature(token_ids=token_ids,
                                 attention_masks=attention_masks,
                                 token_type_ids=token_type_ids,
                                 ent_type=ent2id[_type])
            features.append(feature)

    return features, callback_info


def convert_examples_to_features(examples, max_seq_len, bert_dir, ent2id):

    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))
    features = []
    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')
    type2id = {x: i for i, x in enumerate(ENTITY_TYPES)}

    for i, example in enumerate(examples):
        feature, tmp_callback = convert_mrc_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                ent2id=type2id,
                ent2query=ent2id,
                tokenizer=tokenizer
            )

        if feature is None:
            continue

        features.extend(feature)
        callback_info.extend(tmp_callback)

    logger.info(f'Build {len(features)} features')

    out = (features, )

    if not len(callback_info):
        return out

    type_weight = {}  # 统计每一类的比例，用于计算 micro-f1
    for _type in ENTITY_TYPES:
        type_weight[_type] = 0.

    count = 0.

    for _callback in callback_info:
        type_weight[_callback[-2]] += len(_callback[-1])
        count += len(_callback[-1])

    for key in type_weight:
        type_weight[key] /= count

    out += ((callback_info, type_weight), )

    return out


if __name__ == '__main__':
    pass
