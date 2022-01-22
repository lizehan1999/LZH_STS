import torch
import csv
import os
import random
import time
import numpy as np
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score

random.seed(2022)
np.random.seed(2022)


class Config:
    def __init__(self):
        self.name = 'Sentence_bert_cls'
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
        if flag == 'train':  # 有监督训练集，构造正负例
            data_set = [[self.text2id(t1), self.text2id(t2), l] for t1, t2, l in self.data]
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
        self.fc = nn.Linear(3 * 512, 2)  # 3n*k 的权重矩阵
        self.softmax = nn.Softmax()

    def forward(self, input_ids_t1, attention_mask_t1, token_type_ids_t1, input_ids_t2, attention_mask_t2,
                token_type_ids_t2):
        output_t1 = self.bert(input_ids=input_ids_t1, attention_mask=attention_mask_t1,
                              token_type_ids=token_type_ids_t1)

        output_t2 = self.bert(input_ids=input_ids_t2, attention_mask=attention_mask_t2,
                              token_type_ids=token_type_ids_t2)

        if self.pooling_type == 'cls':
            output_t1 = output_t1[0][:, 0, :]  # (B, d_h)
            output_t2 = output_t2[0][:, 0, :]  # (B, d_h)
        elif self.pooling_type == 'mean':
            output_t1 = torch.mean(output_t1[0], dim=1)  # (B, d_h)
            output_t2 = torch.mean(output_t2[0], dim=1)  # (B, d_h)
        elif self.pooling_type == 'max':
            output_t1 = torch.max(output_t1, dim=1).values  # (B, d_h)
            output_t2 = torch.max(output_t2, dim=1).values  # (B, d_h)

        output = torch.cat((output_t1, output_t2, torch.abs(output_t1 - output_t2)), dim=1)  # (B, 3*d_h)
        output = self.fc(output)
        output = self.softmax(output)  # (B, k)

        return output


def Sbert_loss(pred, labels):
    # pred.shape: [bs dim * k]
    labels = torch.squeeze(labels, dim=-1)
    criteria = nn.CrossEntropyLoss()
    loss = criteria(pred, labels)

    return loss


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

            pred = model(input_ids_t1, attention_mask_t1, token_type_ids_t1, input_ids_t2, attention_mask_t2,
                         token_type_ids_t2)

            y_pred = torch.argmax(pred, dim=1).detach().cpu().numpy().tolist()
            pred_list_all.extend(y_pred)
            label_list = l.squeeze().cpu().numpy().tolist()
            label_list_all.extend(label_list)

    return f1_score(label_list_all, pred_list_all), accuracy_score(label_list_all, pred_list_all), recall_score(
        label_list_all, pred_list_all)


def train_eval(model, optimizer, train_dataloader, dev_dataloader, config):
    best_score = 0.0
    for epoch in range(config.epochs):
        model.train()
        print("***** Running training epoch {} *****".format(epoch + 1))
        train_loss_sum = 0.0
        start = time.time()
        for step, [t1, t2, l] in enumerate(train_dataloader):
            input_ids_t1 = t1['input_ids'].view(len(t1['input_ids']), -1).to(config.device)
            attention_mask_t1 = t1['attention_mask'].view(len(t1['attention_mask']), -1).to(config.device)
            token_type_ids_t1 = t1['token_type_ids'].view(len(t1['token_type_ids']), -1).to(config.device)

            input_ids_t2 = t2['input_ids'].view(len(t2['input_ids']), -1).to(config.device)
            attention_mask_t2 = t2['attention_mask'].view(len(t2['attention_mask']), -1).to(config.device)
            token_type_ids_t2 = t2['token_type_ids'].view(len(t2['token_type_ids']), -1).to(config.device)

            labels = l.to(config.device)

            pred = model(input_ids_t1, attention_mask_t1, token_type_ids_t1, input_ids_t2, attention_mask_t2,
                         token_type_ids_t2)

            loss = Sbert_loss(pred, labels)
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


if __name__ == '__main__':
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
