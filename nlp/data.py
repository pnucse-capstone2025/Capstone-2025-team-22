import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.sentence = data["sentence"]
        self.keyword = data["keyword"]


    def __len__(self):
        return len(self.sentence)
    
    def __getitem__(self, idx):
        return self.sentence[idx], self.keyword[idx]