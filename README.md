## 预训练模型放在根目录下

|             | F1-score | Accuracy_score | Recall_score | 参数                     |
| ----------- | -------- | -------------- | ------------ | ------------------------ |
| Sbert-cls   | 0.8077   | 0.7880         | 0.8905       | bs:64  lr:2e-5           |
| Sbert-cos   | 0.8251   | 0.7687         | 0.8903       | bs:64  lr:2e-5           |
| Sbert-tri   | 0.7999   | 0.7231         | 0.8950       | bs:64  lr:2e-5           |
| UnsupSimCSE | 0.7307   | 0.6217         | 0.8862       | dlr:0.3  bs:64  lr:2e-5  |
| SupSimCSE   | 0.7914   | 0.7355         | 0.8564       | dlr:0.1  bs:128  lr:5e-5 |

