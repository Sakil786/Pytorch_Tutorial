import torch
import tez
import transformers
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics
import pandas as pd


class BertDataset:
    def __init__(self, texts, targets, max_len=64):
        super.__init__()
        self.texts = texts
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=False
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        resp = {
            "idx": torch.tensor(inputs["inputs"], dtype=torch.long),
            "mask": torch.tensor(inputs["inputs"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["inputs"], dtype=torch.long),
            "targets": torch.tensor(self.targets, dtype=torch.float),
        }
        return resp


class TextModel(tez.Model):
    def __init__(self, num_classes, num_train_steps):
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.bert_drop = nn.Dropout(0.03)
        self.out = nn.Linear(768, num_classes)
        self.num_train_steps = num_train_steps

    def fet_optimizer(self):
        opt = AdamW(self.parameters(), lr=1e-4)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_step

        )
        return sch

    def loss(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def monitor_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = torch.cpu().detach().numpy()
        return {
            "accuracy": metrics.accuracy_score(targets, outputs)
        }

    def forward(self, ids, mask, token_ids, targets=None):
        _, x = self.bert(ids, attention_mask=mask, token_ids=token_ids)
        x = self.bert_drop(x)
        x = self.out(x)
        if targets is not None:
            loss = self.loss(x, targets)
            met = self.monitor_metrics(x, targets)
            return x, loss, met
        return x, 0, {}


def train_model():
    df_train = pd.read_csv(r"D:\Pytorch_Tutorial\TextClassification\data\movie_data_sample.csv")
    df_valid = pd.read_csv(r"D:\Pytorch_Tutorial\TextClassification\data\valid_movie_data_sample.csv")
    train_dataset = BertDataset(df_train.review.values, df_train.sentiment.values)
    valid_dataset = BertDataset(df_valid.review.values, df_valid.sentiment.values)
    num_epochs = 10
    train_bs = 32
    nu_train_steps = int(len(df_train) / 32 * 10)
    model = TextModel(num_classes=1, num_train_steps=nu_train_steps)
    es = tez.callbacks.EarlyStopping(monitor="valid_loss", patience=3, model_path="model.bin")
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        epochs=10,
        train_bs=32,
        callbacks=[es],

    )


if __name__ == "__main__":
    train_model()
