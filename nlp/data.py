import torch
from torch.utils.data import Dataset
import json

class KeywordDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        index = item["index"]
        text = item["text"]
        keyword = item["keyword"]
        return {
            "index" : index,
            "text" : text,
            "keyword" : keyword,
        }
    
class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        index = [[item['index']] for item in batch]
        text = [item['text'] for item in batch]

        keyword_id_list = list()
        for item in batch:
            keyword_list = item["keyword"]
            keyword_ids = [self.tokenizer.encode(keyword)[1:-1] for keyword in keyword_list]
            keyword_id_list.append(keyword_ids)

        text_tokens = self.tokenizer(
            text,
            padding=True,
        )
        text_ids = text_tokens['input_ids']
        text_attention_mask = text_tokens['attention_mask']

        # text_ids: (B, L), keyword_id_list: (B, ?), bio_tags: (B, L)
        # 0 : 'B', 1 : 'I', 2 : 'O'
        def create_bio_tags(text_ids, keyword_id_list):
            bio_tags = [[2] * len(text_ids[0]) for _ in range(len(text_ids))]
            for i, keyword_ids in enumerate(keyword_id_list):
                for keyword_id in keyword_ids:
                    keyword_len = len(keyword_id)
                    for j in range(len(text_ids[0]) - keyword_len + 1):
                        if text_ids[i][j: j + keyword_len] == keyword_id:
                            bio_tags[i][j] = 0
                            for k in range(1, keyword_len):
                                bio_tags[i][j + k] = 1
            return bio_tags
        
        bio_tags = create_bio_tags(text_ids, keyword_id_list)
        
        return (torch.tensor(index), torch.tensor(text_ids), torch.tensor(text_attention_mask), torch.tensor(bio_tags))
    
def load_data(file_path=None):
    assert file_path, "There is no file_path"
    with open(file_path, 'r') as f:
        file = json.load(f)
    data = []
    for i_int, i_str in enumerate(file):
        file[i_str]["index"] = i_int
        data.append(file[i_str])
    return data