import torch
from sklearn.datasets import make_classification


class CustomData:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx, :]
        current_target = self.targets[idx]
        return {
            "x": torch.tensor(current_sample, dtype=torch.float),
            "y": torch.tensor(current_target, dtype=torch.long),
        }


if __name__ == '__main__':
    data,target=make_classification(n_samples=100)
    dataset=CustomData(data,target)
    train_dataloder=torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=2)
    for value in train_dataloder:
        print(value)
        break
