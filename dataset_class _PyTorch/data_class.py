import torch


class CustomDataset:
    # data is our text data,targets is our label
    def __init__(self, data, targets,tokenizer):
        self.data = data
        self.targets = targets
        self.tokenizer=tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        target = self.targets[idx]
        input_ids=tokenizer(text)
        return {
            "text"=torch.tensor(input_ids,dtype=torch.float),
            "target"=torch.tensor(target),
        }
