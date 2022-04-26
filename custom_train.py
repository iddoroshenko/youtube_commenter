import torch
import argparse
import math
import numpy as np
from itertools import chain
from tqdm import tqdm

from transformers import (
    SchedulerType,
    get_scheduler,
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
)

import dataset as ds
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from IPython.display import clear_output

set_seed(42)

parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
parser.add_argument("--text_column_name",               default='text')
parser.add_argument("--model_name_or_path",             default='gpt2')
parser.add_argument("--max_train_steps",                default=1_000_000)
parser.add_argument("--block_size",                     default=256, help="Optional input sequence length after tokenization")
parser.add_argument("--log_loss_interval",              default=25)
parser.add_argument("--device",                         default='cuda:0')
parser.add_argument("--num_train_epochs",               default=1)
parser.add_argument("--num_warmup_steps",               default=300)
parser.add_argument("--lr_scheduler_type", type=SchedulerType, default='linear', choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
parser.add_argument("--learning_rate",                  default=5e-5)
parser.add_argument("--output_dir",                     default=None)
args = parser.parse_args(args=[])


def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples['input_ids'])
    if total_length >= args.block_size:
        total_length = (total_length // args.block_size) * args.block_size
    result = {
        k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def eval(model, dataset):
    model.eval()
    losses = []
    perplexity = 0
    for batch in dataset:
        gpu_data = {}
        for key, value in batch.items():
            if key in ['input_ids', 'attention_mask', 'labels']:
                gpu_data[key] = torch.tensor(value).to(args.device)
        
        outputs = model(**gpu_data, use_cache=False)
        losses.append(outputs.loss)
        del gpu_data   
    try:
        perplexity = math.exp(sum(losses)/len(losses))
    except OverflowError:
        perplexity = float("inf")
    return np.mean(losses), perplexity

def show(train_losses, train_perplexities, test_losses, test_perplexities):
    clear_output(wait=True)
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(train_losses, label='train')
    axs[0].plot(test_losses, label='test')
    axs[0].title('Loss')
    axs[1].plot(train_perplexities, label='train')
    axs[1].plot(test_perplexities, label='test')
    axs[0].title('Perplexity')
    plt.show()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path))
    model.resize_token_embeddings(len(tokenizer))

    train, test = ds.load_train_test_dataset(comments_path='data/UScomments.csv', 
                                            videos_path='data/USvideos.csv', 
                                            split=0.8, 
                                            remove_extras=True,
                                            to_lower_case=True,
                                            remove_short_words=False,
                                            max_len=256)
    print("Train:", train)
    print("Test:", test)

    encoded_train = train.map(tokenize_function, batched=True, remove_columns=["text"]).shuffle(seed=42)
    encoded_test = test.map(tokenize_function, batched=True, remove_columns=["text"]).shuffle(seed=42)

    train_dataset = encoded_train.map(group_texts, batched=True)
    test_dataset = encoded_test.map(group_texts, batched=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    progress_bar = tqdm(range(args.max_train_steps), disable=False)
    completed_steps = 0

    test_losses, test_perplexities = [], []
    train_losses, train_perplexities = [], []

    for epoch in range(args.num_train_epochs):
        losses = []
        for step, batch in enumerate(train_dataset):
            model.train()
            gpu_data = {}
            for key, value in batch.items():
                if key in ['input_ids', 'attention_mask', 'labels']:
                    gpu_data[key] = torch.tensor(value).to(args.device)
            
            outputs = model(**gpu_data, use_cache=False)
            loss = outputs.loss
            losses.append(loss.item())
            loss.backward()
            del gpu_data

            progress_bar.update(1)
            completed_steps += 1
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % args.log_loss_interval == 0 and step:
                test_loss, test_perplexity = eval(model, test_dataset)
                test_losses.append(test_loss)
                test_perplexities.append(test_perplexity)

                try:
                    train_perplexity = math.exp(sum(losses)/len(losses))
                except OverflowError:
                    train_perplexity = float("inf")
                train_loss = np.mean(losses)
                train_losses.append(train_loss)
                train_perplexities.append(train_perplexity)
                losses = []

                show(train_losses, train_perplexities, test_losses, test_perplexities)
                raise
                
            if completed_steps >= args.max_train_steps:
                break
