import torch
import csv
import os
import random
import time
import numpy as np
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

random.seed(2022)
np.random.seed(2022)


class Config:
    def __init__(self):
        self.name = 'Sentence_bert_tri'
        self.bert_path = 'chinese_roberta_L-4_H-512'
        self.data_path = 'data'  # 1相似，0不相似
        self.train_path = os.path.join(self.data_path, 'train.tsv')
        self.dev_path = os.path.join(self.data_path, 'dev.tsv')
        self.test_path = os.path.join(self.data_path, 'test.tsv')
        self.save_path = 'save_model.ckpt'

        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.epochs = 1
        self.batch_size = 64
        self.learning_rate = 2e-5
        self.weight_decay = 1e-4
        self.warmup = 0.1
        self.max_length = 30
        self.triplet_margin = 5
        self.pooling_type = 'mean'

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path)


def data_process(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            t1, t2, l = ''.join(row).split('\t')
            data.append([t1, t2, int(l)])
    return data


class SentenceBert(Dataset):
    def __init__(self, data, config, flag):
        self.data = data
        self.tokenizer = config.tokenizer
        self.max_len = config.max_length
        self.data_set = self.sample_loader(flag)

    def sample_loader(self, flag):
        data_set = []
        if flag == 'train':  # 构造正负例
            for t1, t2, l in self.data:
                index = random.randint(0, len(self.data) - 1)
                if l == 1:
                    data_set.append([self.text2id(t1), self.text2id(t2), self.text2id(self.data[index][0]), l])
                elif l == 0:
                    data_set.append([self.text2id(t1), self.text2id(t1), self.text2id(t2), l])
        elif flag == 'dev':  # 验证集
            data_set = [[self.text2id(t1), self.text2id(t2), l] for t1, t2, l in self.data]
        elif flag == 'test':  # 测试集
            data_set = [[self.text2id(t1), self.text2id(t2), l] for t1, t2, l in self.data]
        return data_set

    def text2id(self, text):
        encode_text = self.tokenizer(text=text, max_length=self.max_len, padding='max_length', truncation=True,
                                     return_tensors='pt')
        return encode_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data_set[index]


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert_config = config.bert_config
        self.bert = BertModel.from_pretrained(config.bert_path, config=self.bert_config)
        self.pooling_type = config.pooling_type

    def forward(self, input_ids_t1, attention_mask_t1, token_type_ids_t1, input_ids_t2, attention_mask_t2,
                token_type_ids_t2, input_ids_t3, attention_mask_t3,
                token_type_ids_t3):
        output_t1 = self.bert(input_ids=input_ids_t1, attention_mask=attention_mask_t1,
                              token_type_ids=token_type_ids_t1)

        output_t2 = self.bert(input_ids=input_ids_t2, attention_mask=attention_mask_t2,
                              token_type_ids=token_type_ids_t2)

        output_t3 = self.bert(input_ids=input_ids_t3, attention_mask=attention_mask_t3,
                              token_type_ids=token_type_ids_t3)

        if self.pooling_type == 'cls':
            output_t1 = output_t1[0][:, 0, :]  # (B, d_h)
            output_t2 = output_t2[0][:, 0, :]  # (B, d_h)
            output_t3 = output_t3[0][:, 0, :]  # (B, d_h)
        elif self.pooling_type == 'mean':
            output_t1 = torch.mean(output_t1[0], dim=1)  # (B, d_h)
            output_t2 = torch.mean(output_t2[0], dim=1)  # (B, d_h)
            output_t3 = torch.mean(output_t3[0], dim=1)  # (B, d_h)
        elif self.pooling_type == 'max':
            output_t1 = torch.max(output_t1, dim=1).values  # (B, d_h)
            output_t2 = torch.max(output_t2, dim=1).values  # (B, d_h)
            output_t3 = torch.max(output_t3, dim=1).values  # (B, d_h)

        return output_t1, output_t2, output_t3


def Sbert_loss(pred_t1, pred_t2, pred_t3, triplet_margin: float = 5):
    # pred.shape: [bs dim * k]
    distance_pos = F.pairwise_distance(pred_t1, pred_t2, p=2)
    distance_neg = F.pairwise_distance(pred_t1, pred_t3, p=2)
    losses = F.relu(distance_pos - distance_neg + triplet_margin)

    return losses.mean()


def find_best_f1_and_threshold(labels, scores):
    # copy from sentence_transformers.evaluation.BinaryClassificationEvaluator
    assert len(scores) == len(labels)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=True)

    best_f1 = best_precision = best_recall = 0
    nextract = 0
    ncorrect = 0
    total_num_duplicates = sum(labels)

    for i in range(len(rows) - 1):
        score, label = rows[i]
        nextract += 1

        if label == 1:
            ncorrect += 1

        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / total_num_duplicates
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall

    return best_f1, best_precision, best_recall


def evaluate_test(model, dataloader, config):
    model.eval()
    label_list_all = []
    pred_list_all = []
    with torch.no_grad():
        for index, [t1, t2, l] in enumerate(dataloader):
            input_ids_t1 = t1['input_ids'].view(len(t1['input_ids']), -1).to(config.device)
            attention_mask_t1 = t1['attention_mask'].view(len(t1['attention_mask']), -1).to(config.device)
            token_type_ids_t1 = t1['token_type_ids'].view(len(t1['token_type_ids']), -1).to(config.device)

            input_ids_t2 = t2['input_ids'].view(len(t2['input_ids']), -1).to(config.device)
            attention_mask_t2 = t2['attention_mask'].view(len(t2['attention_mask']), -1).to(config.device)
            token_type_ids_t2 = t2['token_type_ids'].view(len(t2['token_type_ids']), -1).to(config.device)

            input_ids_t3 = t1['input_ids'].view(len(t1['input_ids']), -1).to(config.device)
            attention_mask_t3 = t1['attention_mask'].view(len(t1['attention_mask']), -1).to(config.device)
            token_type_ids_t3 = t1['token_type_ids'].view(len(t1['token_type_ids']), -1).to(config.device)

            pred_t1, pred_t2, pred_t3 = model(input_ids_t1, attention_mask_t1, token_type_ids_t1, input_ids_t2,
                                              attention_mask_t2, token_type_ids_t2, input_ids_t3, attention_mask_t3,
                                              token_type_ids_t3)

            cosine_score = torch.cosine_similarity(pred_t1, pred_t2).detach().cpu().numpy().tolist()
            pred_list_all.extend(cosine_score)
            label_list = l.squeeze().cpu().numpy().tolist()
            label_list_all.extend(label_list)

    return find_best_f1_and_threshold(label_list_all, pred_list_all)


def train_eval(model, optimizer, train_dataloader, dev_dataloader, config):
    best_score = 0.0
    for epoch in range(config.epochs):
        model.train()
        print("***** Running training epoch {} *****".format(epoch + 1))
        train_loss_sum = 0.0
        start = time.time()
        for step, [t1, t2, t3, l] in enumerate(train_dataloader):
            input_ids_t1 = t1['input_ids'].view(len(t1['input_ids']), -1).to(config.device)
            attention_mask_t1 = t1['attention_mask'].view(len(t1['attention_mask']), -1).to(config.device)
            token_type_ids_t1 = t1['token_type_ids'].view(len(t1['token_type_ids']), -1).to(config.device)

            input_ids_t2 = t2['input_ids'].view(len(t2['input_ids']), -1).to(config.device)
            attention_mask_t2 = t2['attention_mask'].view(len(t2['attention_mask']), -1).to(config.device)
            token_type_ids_t2 = t2['token_type_ids'].view(len(t2['token_type_ids']), -1).to(config.device)

            input_ids_t3 = t3['input_ids'].view(len(t3['input_ids']), -1).to(config.device)
            attention_mask_t3 = t3['attention_mask'].view(len(t3['attention_mask']), -1).to(config.device)
            token_type_ids_t3 = t3['token_type_ids'].view(len(t3['token_type_ids']), -1).to(config.device)

            labels = l.to(config.device)

            pred_t1, pred_t2, pred_t3 = model(input_ids_t1, attention_mask_t1, token_type_ids_t1, input_ids_t2,
                                              attention_mask_t2, token_type_ids_t2, input_ids_t3, attention_mask_t3,
                                              token_type_ids_t3)

            loss = Sbert_loss(pred_t1, pred_t2, pred_t3, triplet_margin=config.triplet_margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            train_loss_sum += loss.item()
            if (step + 1) % (len(train_dataloader) // 10) == 0:  # 只打印十次结果
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                    epoch + 1, step + 1, len(train_dataloader), train_loss_sum / (step + 1), time.time() - start))
                model.eval()
                f1_score_, accuracy_score_, recall_score_ = evaluate_test(model, dev_dataloader, config)

                if f1_score_ > best_score:
                    best_score = f1_score_
                    torch.save(model.state_dict(), config.save_path)
                print("current f1 is {:.4f}, acc is {:.4f}, recall is {:.4f}, best f1 is {:.4f}".format(f1_score_,
                                                                                                        accuracy_score_,
                                                                                                        recall_score_,
                                                                                                        best_score))
                print("time costed = {}s \n".format(round(time.time() - start, 5)))

                model.train()


if __name__ == "__main__":
    all_config = Config()

    train_dev_data = data_process(all_config.train_path)
    random.shuffle(train_dev_data)
    train_data = train_dev_data[:int(len(train_dev_data) * 0.9)]
    dev_data = train_dev_data[int(len(train_dev_data) * 0.9):]
    test_data = data_process(all_config.dev_path)
    random.shuffle(test_data)

    train_dataloader = DataLoader(SentenceBert(data=train_data, config=all_config, flag='train'),
                                  batch_size=all_config.batch_size)
    dev_dataloader = DataLoader(SentenceBert(data=dev_data, config=all_config, flag='dev'),
                                batch_size=all_config.batch_size)
    test_dataloader = DataLoader(SentenceBert(data=test_data, config=all_config, flag='test'),
                                 batch_size=all_config.batch_size)

    model = Model(config=all_config).to(all_config.device)
    optimizer = AdamW(model.parameters(), lr=all_config.learning_rate)
    total_steps = int(len(train_dataloader) * all_config.epochs)

    train_eval(model, optimizer, train_dataloader, dev_dataloader, all_config)
    f1, acc, recall = evaluate_test(model, test_dataloader, all_config)  # test
    print("f1 is {:.4f}, acc is {:.4f}, recall is {:.4f}".format(f1, acc, recall))
