import torch
from torch import nn
from transformers import BertModel, BertConfig
from TorchCRF import CRF
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
#from app.nlp.utils.extract import extract_keywords_from_bio_tags
from ..utils.extract import extract_keywords_from_bio_tags

class KoKeyBERT(nn.Module):
    """ Korean KeyBERT Model """
    def __init__(self, 
                config:BertConfig=None, 
                num_class:int = 3, 
                model_name:str = 'skt/kobert-base-v1')->None:
        
        super().__init__()
        
        if config is None:
            self.model = BertModel.from_pretrained(model_name)
            self.config = self.model.config
        else:
            self.config = config
            self.model = BertModel(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_class)
        self.crf = CRF(num_labels = num_class)

    def forward(self,
                input_ids:torch.LongTensor,
                attention_mask:Optional[torch.FloatTensor] = None,
                tags:Optional[torch.LongTensor] = None,
                return_outputs:Optional[bool] = False
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
        emissions = emissions.float()
        mask = attention_mask.to(torch.bool)


        if tags is not None:
            labels = tags.long()
            log_likelihood, sequence_of_tags = self.crf(emissions, labels, mask) , self.crf.viterbi_decode(emissions, mask)
            if return_outputs:
                return log_likelihood, sequence_of_tags, outputs
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.viterbi_decode(emissions, mask)
            if return_outputs:
                return sequence_of_tags, outputs
            return sequence_of_tags
        
    def extract_keywords(self,
                         text,
                         tokenizer,
                         ) -> set:
        """
        input: Natural language text
        """
        input = tokenizer(text, return_tensors='pt')

        input_ids = input['input_ids'].to(self.device)  
        attention_mask = input['attention_mask'].to(self.device)
        
        outputs = self.forward(input_ids = input_ids,
                            attention_mask = attention_mask
                            )
        
        padded = torch.ones_like(input_ids, dtype=torch.long)
        text_ids = tokenizer.encode(text)
        pred_keywords = []
        try:
            # 예측된 키워드 추출
            pred_keyword = extract_keywords_from_bio_tags(
                text_ids,
                outputs[0],
                attention_mask[0],
                tokenizer,
                self.device
            )
            pred_keywords.append(pred_keyword)
        except Exception as e:
            print(e)
        
        pred_keywords = set(pred_keywords[0])
        if pred_keywords:
            return pred_keywords
        else:
            return None
    
if __name__ == '__main__':
    from transformers import BertConfig
    from ..tokenizer.kobert_tokenizer import KoBERTTokenizer

    # Model Loading Part

    # Load the model configuration
    config = BertConfig.from_pretrained('skt/kobert-base-v1')
    model = KoKeyBERT(config=config)

    # Load the model state
    model.load_state_dict(torch.load('src/model_state/best_model.pt', map_location=torch.device('cpu'), weights_only=True))

    # Load the tokenizer
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')


    while True:
        text = input('Enter a text: ')
        print(model.extract_keywords(text, tokenizer))
