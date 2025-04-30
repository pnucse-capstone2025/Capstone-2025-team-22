import torch
from torch import nn
from transformers import BertModel, BertConfig
from TorchCRF import CRF
from typing import Optional

class KoKeyBERT(nn.Module):
    """ Korean KeyBERT Model """
    def __init__(self, 
                config:BertConfig, 
                num_class:int = 3, 
                model_name:str = 'skt/kobert-base-v1')->None:
        
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        if config is None:
            self.config = model.config
        else:
            self.config = config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_class)
        self.crf = CRF(num_labels = num_class)

    def forward(self,
                input_ids:torch.LongTensor,
                attention_mask:Optional[torch.FloatTensor] = None,
                tags:Optional[torch.LongTensor] = None
                ):
        """
        input: (B, L)
        """

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).float()

        outputs = self.model(input_ids = input_ids, 
                            attention_mask = attention_mask,
                            )
        
        # (B, L, H)
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        
        # (B, L, num_class)
        emissions = self.classifier(last_hidden_state)

        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, tags) , self.crf.viterbi_decode(emissions)
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf(emissions, tags)
            return sequence_of_tags

