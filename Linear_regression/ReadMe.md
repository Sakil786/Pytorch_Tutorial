### Introduction
In this repository , we'll dive into the code for creating a simple linear regression model using PyTorch. We'll construct a custom dataset, split it into training and testing sets, and train the model to perform simple regression tasks. Let's break down the code line by line to understand each step.

# Line-by-Line Explanation:
```python
1. import torch
2. from sklearn.datasets import make_classification
3. from sklearn.model_selection import train_test_split


```
We begin by importing the necessary libraries, including PyTorch for tensor operations and scikit-learn for dataset creation and manipulation.

```python
5. class CustomeDataset:
6.     def __init__(self, data, targets):
7.         self.data = data
8.         self.targets = targets
9.     def __len__(self):
10.         return self.data.shape[0]


```
The CustomeDataset class is defined to encapsulate our custom dataset. It is initialized with input features (data) and corresponding target labels (targets). The __len__ method returns the number of samples in the dataset.

```python
12.     def __getitem__(self, idx):
13.         custom_sample = self.data[idx, :]
14.         custom_target = self.targets[idx]
15.         return {
16.             "x": torch.tensor(custom_sample, dtype=torch.float),
17.             "y": torch.tensor(custom_target, dtype=torch.long),
18.         }



```
The __getitem__ method retrieves a sample at the given index (idx). It fetches the current input sample and its corresponding target label, returning a dictionary containing the input features ("x") and the target label ("y") as PyTorch tensors.

```python
20. if __name__ == '__main__':
21.     data, target = make_classification(n_samples=1000)
22.     train_data, test_data, train_target, test_target = train_test_split(data, target, stratify=target)
23.     train_dataset = CustomeDataset(train_data, train_target)
24.     test_dataset = CustomeDataset(test_data, test_target)


```
In the main block, we generate a synthetic dataset using make_classification and split it into training and testing sets using train_test_split. Instances of the CustomeDataset class are then created for both training and testing.

```python
26.     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=2)
27.     test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, num_workers=2)

```
We create PyTorch DataLoader instances for both the training and testing datasets, specifying a batch size of 4 and utilizing 2 worker processes for efficient data loading.
```python
29.     model = lambda x, w, b: torch.matmul(x, w) + b
30.     w = torch.randn(20, 1, requires_grad=True)
31.     b = torch.randn(1, requires_grad=True)

```
We define a simple linear regression model represented by the model lambda function. Parameters w and b are initialized with random values and marked for gradient computation.
```python
33.     learning_rate = 0.001
34.     for epoch in range(10):
35.         epoch_loss = 0
36.         for data in train_dataloader:
37.             xtrain = data["x"]
38.             ytrain = data["y"]
39.             output = model(xtrain, w, b)
40.             loss = torch.mean((ytrain.view(-1) - output.view(-1))**2)
41.             epoch_loss = epoch_loss + loss.item()
42.             loss.backward()
43.             with torch.no_grad():
44.                 w = w - learning_rate * w.grad
45.                 b = b - learning_rate * b.grad
46.             w.requires_grad_(True)
47.             b.requires_grad_(True)
48.         print(epoch, epoch_loss)

```
Finally, we train the model using a simple gradient descent approach. We iterate through epochs and mini-batches, computing the mean squared error loss, backpropagating gradients, and updating the model parameters.
