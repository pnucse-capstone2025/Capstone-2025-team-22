from data import KeywordDataset, Collator, load_data
from model import KoKeyBERT
from kobert_tokenizer import KoBERTTokenizer

import argparse
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.optimizer import Optimizer
from transformers import get_linear_schedule_with_warmup
import torch
import os
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def train(model: KoKeyBERT,
          optimizer: Optimizer,
          scheduler: get_linear_schedule_with_warmup,
          train_dataset: KeywordDataset,
          val_dataset: KeywordDataset,
          epoch:int = 0,
          step:int = 0,
          num_epochs:int = 12,
          learning_rate:float = 5e-5,
          random_seed:int = 42,
          collator:Collator = None,
          best_val_loss:float = float('inf'),
          best_val_acc:float = 0.0,
          device:torch.device = None,
          args:argparse.Namespace = None,
          ):

    logger = logging.getLogger(args.train_logger_name)
    logger.setLevel(logging.DEBUG)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # file handler
    file_handler = logging.FileHandler("./log/" + args.train_logger_name + ".log")
    file_handler.setLevel(logging.DEBUG)

    # format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info("********** Start training **********")
    logger.info("Num epochs: %d", num_epochs)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Learning rate: %f", learning_rate)
    logger.info("Random seed: %d", random_seed)
    
    train_dataloader = DataLoader(train_dataset, 
                                   batch_size=args.batch_size, 
                                   collate_fn=collator, 
                                   sampler=RandomSampler(train_dataset),
                                   num_workers=args.num_workers)

    model.train()
    while epoch < num_epochs:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            model.zero_grad()
            # batch
            index, text_ids, text_attention_mask, bio_tags = batch

            # return log_likelihood, sequence_of_tags
            train_loss, sequence_of_tags = model(input_ids=text_ids.to(device), 
                                                attention_mask=text_attention_mask.to(device), 
                                                tags=bio_tags.to(device), 
                                                return_outputs=False)
            # Ensure train_loss is a scalar
            train_loss = -train_loss.mean()
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                tag_seqs = [torch.tensor(s, dtype=torch.long) for s in sequence_of_tags]
                # 배치 내 가장 긴 시퀀스 길이로 패딩
                padded = pad_sequence(tag_seqs, batch_first=True, padding_value=model.config.pad_token_id).to(device)
                mb_acc = (padded == bio_tags.to(device)).float()[text_attention_mask.bool()].mean()
                

            tr_acc = mb_acc.item()
            logger.info("Step: %d, Loss: %f, Acc: %f", step, train_loss.item(), tr_acc)

            if args.val_freq is not None and step % args.val_freq == 0:
                logger.info("Validating...")
                val_loss, val_acc = evaluate(model, val_dataset, collator, args, logger, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    logger.info("Best val loss: %f, Best val acc: %f", best_val_loss, best_val_acc)
                    # save model
                    os.makedirs("./checkpoints", exist_ok=True)
                    logger.info("Saving best model to ./checkpoints/best_model.pt")
                    torch.save(model.state_dict(), os.path.join("./checkpoints", "best_model.pt"))
            
            if args.save_freq is not None and step % args.save_freq == 0:
                # save current training state in one file
                logger.info("Saving current training state to ./checkpoints/step_" + str(step) + "_state.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                }, os.path.join("./checkpoints", "step_" + str(step) + "_state.pt"))

def evaluate(model: KoKeyBERT,
             val_dataset: KeywordDataset,
             collator:Collator = None,
             args:argparse.Namespace = None,
             logger:logging.Logger = None,
             device:torch.device = None,
             ):

    sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=args.batch_size, 
                                collate_fn=collator, 
                                sampler=sampler,
                                num_workers=args.num_workers,
                                drop_last=False)
    model.eval()
    total_val_loss = 0.0
    total_val_acc = 0.0
    val_step = 0
    with torch.no_grad():
        for batch in val_dataloader:
            val_step += 1
            index, text_ids, text_attention_mask, bio_tags = batch

            # return log_likelihood, sequence_of_tags
            log_likelihood, sequence_of_tags = model(input_ids=text_ids.to(device), 
                                                    attention_mask=text_attention_mask.to(device), 
                                                    tags=bio_tags.to(device), 
                                                    return_outputs=False)
            log_likelihood = -log_likelihood.mean()
            total_val_loss += log_likelihood.item()
            
            # Handle sequence_of_tags as in training
            tag_seqs = [torch.tensor(s, dtype=torch.long) for s in sequence_of_tags]
            padded = pad_sequence(tag_seqs, batch_first=True, padding_value=model.config.pad_token_id).to(device)
            mb_acc = (padded == bio_tags.to(device)).float()[text_attention_mask.bool()].mean()
            
            total_val_acc += mb_acc.item()
            current_val_loss = log_likelihood.item()
            current_val_acc = mb_acc.item()
            
            logger.info("Current Step: %d, Val loss: %f, Acc: %f", val_step, current_val_loss, current_val_acc)
            
    
    total_val_loss /= len(val_dataloader)
    total_val_acc /= len(val_dataloader)
    logger.info("Total Step: %d, Total Val loss: %f, Acc: %f", val_step, total_val_loss, total_val_acc)
    return total_val_loss, total_val_acc
    
def main():
    args = argparse.ArgumentParser(description='KoKeyBERT Training')
    args.add_argument("--train_data_path", type=str, default="./src/data/train_clean.json")
    args.add_argument("--model_name", type=str, default="skt/kobert-base-v1")
    args.add_argument("--num_epochs", type=int, default=12)
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--learning_rate", type=float, default=5e-5)
    args.add_argument("--random_seed", type=int, default=42)
    args.add_argument("--split_ratio", type=float, default=0.2)
    args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="xla for tpu, cuda for gpu, cpu for cpu")
    args.add_argument("--train_logger_name", type=str, default="train")
    args.add_argument("--num_workers", type=int, default=8, help="A100: 12, 8 recommanded")
    args.add_argument("--num_warmup_steps", type=int, default=100)
    args.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training from")
    args.add_argument("--val_freq", type=int, default=100)
    args.add_argument("--save_freq", type=int, default=100)
    args = args.parse_args()

    # Set random seed
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.random_seed)
        device = torch.device("cuda")
    elif args.device == "cpu":
        torch.manual_seed(args.random_seed)
        device = torch.device("cpu")
    elif args.device == "xla":
        import torch_xla.core.xla_model as xm
        xm.set_rng_seed(args.random_seed)
        device = xm.xla_device()
    else:
        raise ValueError(f"Unknown device: {args.device}")
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Create necessary directories
    os.makedirs("./log", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    # Load and split data
    train_data = load_data(args.train_data_path)
    if train_data is None:
        raise ValueError(f"Failed to load data from {args.train_data_path}")

    train_data, val_data = train_test_split(train_data, random_state=args.random_seed, test_size=args.split_ratio)

    train_dataset = KeywordDataset(train_data)
    val_dataset = KeywordDataset(val_data)

    # Initialize training state
    step = 0
    epoch = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0    
    
    # Initialize tokenizer and collator
    tokenizer = KoBERTTokenizer.from_pretrained(args.model_name)
    if tokenizer is None:
        raise ValueError(f"Failed to load tokenizer for model {args.model_name}")
    collator = Collator(tokenizer)

    # Load checkpoint if provided
    if args.checkpoint_path is not None:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        try:
            checkpoint = torch.load(args.checkpoint_path)
            
            # Load model
            model = KoKeyBERT()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
        
        # Load optimizer and scheduler
        param_optimizer = list(model.model.named_parameters()) \
                        + list(model.classifier.named_parameters()) \
                        + list(model.crf.named_parameters())
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]    
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_epochs * (len(train_dataset) // args.batch_size)
        )
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        step = checkpoint['step']
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint['best_val_acc']
        
        print(f"Checkpoint loaded. Resuming from epoch {epoch}, step {step}")
    else:
        # Initialize new training
        model = KoKeyBERT()
        model.to(device)
        
        param_optimizer = list(model.model.named_parameters()) \
                        + list(model.classifier.named_parameters()) \
                        + list(model.crf.named_parameters())
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]    
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_epochs * (len(train_dataset) // args.batch_size)
        )


    train(model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          train_dataset=train_dataset,
          val_dataset=val_dataset,
          step=step,
          epoch=epoch,
          num_epochs=args.num_epochs,
          learning_rate=args.learning_rate,
          random_seed=args.random_seed,
          collator=collator,
          best_val_loss=best_val_loss,
          best_val_acc=best_val_acc,
          device=device,
          args=args,
          )

if __name__ == "__main__":
    main()