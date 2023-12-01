### Introduction
This blog post will guide you through the process of creating a PyTorch DataLoader for a custom dataset generated with make_classification from scikit-learn. Let's dissect the code line by line to understand its functionality.

# Line-by-Line Explanation:
```python
1. import torch
2. from sklearn.datasets import make_classification

```
Here, we import the necessary libraries: torch for PyTorch and make_classification from scikit-learn to generate synthetic data.

```python
4. class CustomData:
5.     def __init__(self, data, targets):
6.         self.data = data
7.         self.targets = targets
8.     def __len__(self):
9.         return len(self.data)

```
The CustomData class is defined to encapsulate the custom dataset. The __init__ method initializes the dataset with input features (data) and corresponding target labels (targets). The __len__ method returns the length of the dataset.

```python
11.     def __getitem__(self, idx):
12.         current_sample = self.data[idx, :]
13.         current_target = self.targets[idx]
14.         return {
15.             "x": torch.tensor(current_sample, dtype=torch.float),
16.             "y": torch.tensor(current_target, dtype=torch.long),
17.         }


```
The __getitem__ method retrieves a sample at the given index (idx). It fetches the current input sample and its corresponding target label. It returns a dictionary containing the input features ("x") and the target label ("y") as PyTorch tensors.

```python
19. if __name__ == '__main__':
20.     data, target = make_classification(n_samples=100)
21.     dataset = CustomData(data, target)
22.     train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
23.     for value in train_dataloader:
24.         print(value)
25.         break

```
In the main block, we generate synthetic data using make_classification and create an instance of the CustomData class. We then use torch.utils.data.DataLoader to create a DataLoader, specifying a batch size of 4 and 2 worker processes for data loading efficiency. The loop iterates over the DataLoader to print the first batch.
