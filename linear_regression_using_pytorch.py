import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class CustomeDataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        custom_sample = self.data[idx, :]
        custom_target = self.targets[idx]

        return {
            "x": torch.tensor(custom_sample, dtype=torch.float),
            "y": torch.tensor(custom_target, dtype=torch.long),
        }


if __name__ == '__main__':
    data, target = make_classification(n_samples=1000)
    train_data, test_data, train_target, test_target = train_test_split(data, target, stratify=target)
    train_dataset = CustomeDataset(train_data, train_target)
    test_dataset = CustomeDataset(test_data, test_target)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, num_workers=2)
    model = lambda x, w, b: torch.matmul(x, w) + b
    w = torch.randn(20, 1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    learning_rate = 0.001
    for epoch in range(10):
        epoch_loss = 0
        for data in train_dataloader:
            xtrain = data["x"]
            ytrain = data["y"]
            output = model(xtrain, w, b)
            loss = torch.mean((ytrain.view(-1) - output.view(-1))**2)
            epoch_loss = epoch_loss + loss.item()
            loss.backward()
            with torch.no_grad():
                w = w - learning_rate * w.grad
                b = b - learning_rate * b.grad
            w.requires_grad_(True)
            b.requires_grad_(True)
        print(epoch, epoch_loss)
