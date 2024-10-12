# Author: wormiz
import torch
import torch.utils

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], torch.as_tensor(self.labels[idx]).item()
    