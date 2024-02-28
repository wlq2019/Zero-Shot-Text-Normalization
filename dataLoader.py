# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     dataLoader
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/6/26
   Software:      PyCharm
'''
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

'''
Dataset:(Dataset)
    1. __init__()
    2. __len__()
    3. __getitem__()

Dataloader:
    1. collate_fn()
'''

class PunctuationDataset(Dataset):
    def __init__(self, input_path, label_vocab_path):
        with open(input_path, "r", encoding="utf-8") as f:
            sentence_label_pair = [line.strip().split("|| ") for line in f.readlines()]

        self.inputs = sentence_label_pair
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # use offline pretrained bert
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.label_vocab = self._read_dict(label_vocab_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sentence = self.inputs[index][0]
        label = self.inputs[index][1]
        # label = 'O' # 单语语料的蒸馏训练
        label = label.replace("B-SELF","O").replace("M-SELF","O").replace("E-SELF","O").replace("S-SELF","O").replace("S-POWER_OPERATOR","O").replace("B-POINT","S-POINT").replace("M-POINT","S-POINT").replace("E-POINT","S-POINT").replace("B-SLASH_PER","S-SLASH_PER").replace("M-SLASH_PER","S-SLASH_PER").replace("E-SLASH_PER","S-SLASH_PER").replace("B-COLON_HOUR","S-COLON_HOUR").replace("M-COLON_HOUR","S-COLON_HOUR").replace("E-COLON_HOUR","S-COLON_HOUR").replace("B-SLASH_FRACTION","S-SLASH_FRACTION").replace("M-SLASH_FRACTION","S-SLASH_FRACTION").replace("E-SLASH_FRACTION","S-SLASH_FRACTION")
        # print(label)
        # Convert to id
        label = self._w2i(self.label_vocab, label)
        sentence = self.tokenizer.convert_tokens_to_ids(sentence.split(" "))

        return sentence, label

    @staticmethod
    def _w2i(vocab, sequence):
        # print(sequence)
        res = [int(vocab.get(w, vocab.get("<UNK>"))) for w in sequence.split(" ")]
        return res


    @staticmethod
    def _read_dict(dict_path):
        with open(dict_path, "r", encoding="utf-8") as dict:
            lines = [line.strip() for line in dict.readlines()]
        res = {e.split("||")[0]: e.split("||")[1] for e in lines}
        return res


class PunctuationDataset_dl(Dataset):
    def __init__(self, input_path, label_vocab_path):
        with open(input_path, "r", encoding="utf-8") as f:
            sentence_label_pair = [line.strip().split("|| ") for line in f.readlines()]

        self.inputs = sentence_label_pair
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # use offline pretrained bert
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.label_vocab = self._read_dict(label_vocab_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sentence = self.inputs[index][0]
        # label = self.inputs[index][1]
        label = 'O' # 单语语料的蒸馏训练
        label = label.replace("B-SELF","O").replace("M-SELF","O").replace("E-SELF","O").replace("S-SELF","O").replace("S-POWER_OPERATOR","O").replace("B-POINT","S-POINT").replace("M-POINT","S-POINT").replace("E-POINT","S-POINT").replace("B-SLASH_PER","S-SLASH_PER").replace("M-SLASH_PER","S-SLASH_PER").replace("E-SLASH_PER","S-SLASH_PER").replace("B-COLON_HOUR","S-COLON_HOUR").replace("M-COLON_HOUR","S-COLON_HOUR").replace("E-COLON_HOUR","S-COLON_HOUR").replace("B-SLASH_FRACTION","S-SLASH_FRACTION").replace("M-SLASH_FRACTION","S-SLASH_FRACTION").replace("E-SLASH_FRACTION","S-SLASH_FRACTION").replace("ordinal","O")
        # print(label)
        # Convert to id
        label = self._w2i(self.label_vocab, label)
        sentence = self.tokenizer.convert_tokens_to_ids(sentence.split(" "))

        return sentence, label

    @staticmethod
    def _w2i(vocab, sequence):
        # print(sequence)
        res = [int(vocab.get(w, vocab.get("<UNK>"))) for w in sequence.split(" ")]
        return res


    @staticmethod
    def _read_dict(dict_path):
        with open(dict_path, "r", encoding="utf-8") as dict:
            lines = [line.strip() for line in dict.readlines()]
        res = {e.split("||")[0]: e.split("||")[1] for e in lines}
        return res



def collate_fn(data):
    sentences, labels= zip(*data)

    # 获取本batch中最长的序列
    sequence_lengths = [len(line) for line in sentences]
    max_length = max(sequence_lengths)

    # 初始化两个结果矩阵，全置0，等待后续迭代替换非0元素，尺寸:batch_size x max_sequence_length
    res_sentences, res_labels = torch.zeros(len(sentences), max_length, dtype=torch.long), torch.zeros(len(labels), max_length, dtype=torch.long)

    # 使用原序列非零元素替换结果矩阵
    for index, sentence_label_pair in enumerate(zip(sentences, labels)):
        real_length = sequence_lengths[index]
        res_sentences[index, :real_length] = torch.LongTensor(sentence_label_pair[0])[:real_length]
        res_labels[index, :real_length] = torch.LongTensor(sentence_label_pair[1])[:real_length]

        assert res_sentences.size() == res_labels.size()

        seq_len = torch.LongTensor(sequence_lengths)

    # return res_sentences, res_labels # for train
    return res_sentences, res_labels, seq_len # for test

# def collate_fn_ml(data):
#     sentences, labels= zip(*data)

#     # 获取本batch中最长的序列
#     sequence_lengths = [len(line) for line in sentences]
#     max_length = max(sequence_lengths)

#     # 初始化两个结果矩阵，全置0，等待后续迭代替换非0元素，尺寸:batch_size x max_sequence_length
#     res_sentences, res_labels = torch.zeros(len(sentences), max_length, dtype=torch.long), torch.zeros(len(labels), max_length, dtype=torch.long)

#     # 使用原序列非零元素替换结果矩阵
#     for index, sentence_label_pair in enumerate(zip(sentences, labels)):
#         real_length = sequence_lengths[index]
#         res_sentences[index, :real_length] = torch.LongTensor(sentence_label_pair[0])[:real_length]
#         res_labels[index, :real_length] = torch.LongTensor(sentence_label_pair[1])[:real_length]

#         assert res_sentences.size() == res_labels.size()

#         seq_len = torch.LongTensor(sequence_lengths)

#     return res_sentences, res_labels # for train
#     # return res_sentences, res_labels, seq_len # for testz






class PunctuationDataset_student(Dataset):
    def __init__(self, input_path, label_vocab_path, preds_all):
        with open(input_path, "r", encoding="utf-8") as f:
            sentence_label_pair = [line.strip().split("|| ") for line in f.readlines()]

        self.inputs = sentence_label_pair
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # use offline pretrained bert
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
       
        self.label_vocab = self._read_dict(label_vocab_path)
        self.soft_label = preds_all
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sentence = self.inputs[index][0]
        soft_label = self.soft_label[index]

        WFST_label = self.inputs[index][1]
        # label = 'O' # 单语语料的蒸馏训练
        WFST_label = WFST_label.replace("B-SELF","O").replace("M-SELF","O").replace("E-SELF","O").replace("S-SELF","O").replace("S-POWER_OPERATOR","O").replace("B-POINT","S-POINT").replace("M-POINT","S-POINT").replace("E-POINT","S-POINT").replace("B-SLASH_PER","S-SLASH_PER").replace("M-SLASH_PER","S-SLASH_PER").replace("E-SLASH_PER","S-SLASH_PER").replace("B-COLON_HOUR","S-COLON_HOUR").replace("M-COLON_HOUR","S-COLON_HOUR").replace("E-COLON_HOUR","S-COLON_HOUR").replace("B-SLASH_FRACTION","S-SLASH_FRACTION").replace("M-SLASH_FRACTION","S-SLASH_FRACTION").replace("E-SLASH_FRACTION","S-SLASH_FRACTION").replace("measure","S-CARDINAL").replace("ordinal","O").replace("money","S-CARDINAL").replace("telephone","S-CARDINAL")
        sentence = self.tokenizer.convert_tokens_to_ids(sentence.split(" "))
        # print(WFST_label)
        WFST_label = self._w2i(self.label_vocab, WFST_label)
        # print(WFST_label)

        return sentence, soft_label, WFST_label
    
    @staticmethod
    def _w2i(vocab, sequence):
        # print(sequence)
        res = [int(vocab.get(w, vocab.get("<UNK>"))) for w in sequence.split(" ")]
        return res


    @staticmethod
    def _read_dict(dict_path):
        with open(dict_path, "r", encoding="utf-8") as dict:
            lines = [line.strip() for line in dict.readlines()]
        res = {e.split("||")[0]: e.split("||")[1] for e in lines}
        return res

   


def collate_fn_student(data):
    sentences, soft_label, WFSt_labels= zip(*data)

    # 获取本batch中最长的序列
    sequence_lengths = [len(line) for line in sentences]
    max_length = max(sequence_lengths)

    # 初始化两个结果矩阵，全置0，等待后续迭代替换非0元素，尺寸:batch_size x max_sequence_length
    res_sentences, res_WFSt_labels = torch.zeros(len(sentences), max_length, dtype=torch.long), torch.zeros(len(WFSt_labels), max_length, dtype=torch.long)


    # 使用原序列非零元素替换结果矩阵
    for index, sentence_label_pair in enumerate(zip(sentences, WFSt_labels)):
        real_length = sequence_lengths[index]
        res_sentences[index, :real_length] = torch.LongTensor(sentence_label_pair[0])[:real_length]
        res_WFSt_labels[index, :real_length] = torch.LongTensor(sentence_label_pair[1])[:real_length]


        seq_len = torch.LongTensor(sequence_lengths)

    soft_label = torch.LongTensor(soft_label)
    return res_sentences, soft_label, res_WFSt_labels # for train



if __name__ == '__main__':
    dataset = PunctuationDataset("../dataset/LibriTTS/processed_for_new/dev-clean.tsv", "../dataset/LibriTTS/processed_for_new/label.dict.tsv")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for sentence, label in tqdm(dataloader):
        pass
        # print(f"sentence: {sentence}; label: {label}\n")
        # break
