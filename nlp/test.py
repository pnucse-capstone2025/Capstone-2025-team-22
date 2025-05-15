from data import KeywordDataset, Collator, load_data
from model import KoKeyBERT
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertConfig

import argparse
import logging
from torch.utils.data import DataLoader, SequentialSampler
import torch
import os
from torch.nn.utils.rnn import pad_sequence

def test(model: KoKeyBERT,
         test_dataset: KeywordDataset,
         collator: Collator = None,
         args: argparse.Namespace = None,
         logger: logging.Logger = None,
         device: torch.device = None,
         ):

    sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=args.batch_size, 
                                collate_fn=collator, 
                                sampler=sampler,
                                num_workers=args.num_workers,
                                drop_last=False)
    
    model.eval()
    total_test_loss = 0.0
    total_test_acc = 0.0
    test_step = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            test_step += 1
            index, text_ids, text_attention_mask, bio_tags = batch

            # return log_likelihood, sequence_of_tags
            log_likelihood, sequence_of_tags = model(input_ids=text_ids.to(device), 
                                                    attention_mask=text_attention_mask.to(device), 
                                                    tags=bio_tags.to(device), 
                                                    return_outputs=False)
            log_likelihood = -log_likelihood.mean()
            total_test_loss += log_likelihood.item()
            
            # Handle sequence_of_tags as in training
            tag_seqs = [torch.tensor(s, dtype=torch.long) for s in sequence_of_tags]
            padded = pad_sequence(tag_seqs, batch_first=True, padding_value=model.config.pad_token_id).to(device)
            mb_acc = (padded == bio_tags.to(device)).float()[text_attention_mask.bool()].mean()
            
            total_test_acc += mb_acc.item()
            current_test_loss = log_likelihood.item()
            current_test_acc = mb_acc.item()
            
            logger.info("Current Step: %d, Test loss: %f, Acc: %f", test_step, current_test_loss, current_test_acc)
            
    
    total_test_loss /= len(test_dataloader)
    total_test_acc /= len(test_dataloader)
    logger.info("Total Step: %d, Total Test loss: %f, Acc: %f", test_step, total_test_loss, total_test_acc)
    return total_test_loss, total_test_acc

def main():
    args = argparse.ArgumentParser(description='KoKeyBERT Testing')
    args.add_argument("--test_data_path", type=str, default="./src/data/test_clean.json")
    args.add_argument("--model_name", type=str, default="skt/kobert-base-v1")
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda for gpu, cpu for cpu")
    args.add_argument("--test_logger_name", type=str, default="test")
    args.add_argument("--num_workers", type=int, default=8, help="A100: 12, 8 recommanded")
    args.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model.pt", help="Path to checkpoint to load model from")
    args = args.parse_args()

    # Set device
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {args.device}")

    # Create necessary directories
    os.makedirs("./log", exist_ok=True)

    # Load test data
    test_data = load_data(args.test_data_path)
    if test_data is None:
        raise ValueError(f"Failed to load data from {args.test_data_path}")

    test_dataset = KeywordDataset(test_data)

    # Initialize logger
    logger = logging.getLogger(args.test_logger_name)
    logger.setLevel(logging.DEBUG)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # file handler
    file_handler = logging.FileHandler("./log/" + args.test_logger_name + ".log")
    file_handler.setLevel(logging.DEBUG)

    # format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Initialize tokenizer and collator
    tokenizer = KoBERTTokenizer.from_pretrained(args.model_name)
    if tokenizer is None:
        raise ValueError(f"Failed to load tokenizer for model {args.model_name}")
    collator = Collator(tokenizer)

    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint_path}")
    try:
        config = BertConfig.from_pretrained(args.model_name)
        model = KoKeyBERT(config=config)
        model.load_state_dict(torch.load(args.checkpoint_path))
        model.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")

    # Run test
    logger.info("********** Start testing **********")
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Device: %s", args.device)
    
    test_loss, test_acc = test(model=model,
                              test_dataset=test_dataset,
                              collator=collator,
                              args=args,
                              logger=logger,
                              device=device)
    
    logger.info("********** Testing finished **********")
    logger.info("Final Test loss: %f, Acc: %f", test_loss, test_acc)

if __name__ == "__main__":
    main()
