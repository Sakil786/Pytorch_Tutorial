### Introduction
This repo  will walk you through a simple yet powerful example of constructing a custom dataset class for text classification using PyTorch. Let's explore the code line by line to understand its functionality.

# Line-by-Line Explanation:
```python
1. import torch
```
This line imports the PyTorch library, a widely-used open-source machine learning framework.

```python
3. class CustomDataset:
```
Here, we define a class named CustomDataset to encapsulate our text data and labels.

```python
4.     def __init__(self, data, targets, tokenizer):
5.         self.data = data
6.         self.targets = targets
7.         self.tokenizer = tokenizer

```
The __init__ method initializes the dataset with the provided data, targets, and a tokenizer. data represents the text data, targets are the corresponding labels, and tokenizer is a function that will be used to tokenize the text.

```python
9.     def __len__(self):
10.         return len(self.data)
```
The __len__ method returns the length of the dataset, indicating the number of samples.

```python
12.     def __getitem__(self, idx):
13.         text = self.data[idx]
14.         target = self.targets[idx]
15.         input_ids = self.tokenizer(text)

```
The __getitem__ method retrieves a sample at the given index (idx). It fetches the text and corresponding target label. It then applies the provided tokenizer to convert the text into tokenized input.
    17.         return {
    18.             "text": torch.tensor(input_ids, dtype=torch.float),
    19.             "target": torch.tensor(target),
    20.         }
    
Finally, the method returns a dictionary containing the tokenized text as a PyTorch tensor ("text") and the target label as another PyTorch tensor ("target").
