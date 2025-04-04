import torch
from torch.utils.data import Dataset
import json

class KeywordDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        datum = self.data[idx]
        text = datum["text"]
        keyword = datum["keyword"]
        return {
            "text" : text,
            "keyword" : keyword,
        }
    
def load_data(file_path=None):
    assert file_path, "There is no file_path"
    with open(file_path, 'r') as f:
        file = json.load(f)
    data = []
    for i in file:
        data.append(file[i])
    return data