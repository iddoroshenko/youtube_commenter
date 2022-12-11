import argparse
import math
from itertools import chain
import dataset as ds
import warnings
from transformers import (
    SchedulerType,
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
warnings.filterwarnings("ignore")
set_seed(42)

parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
parser.add_argument("--text_column_name",               default='text')
parser.add_argument("--model_name_or_path",             default='gpt2')
parser.add_argument("--max_train_steps",                default=1_000_000)
parser.add_argument("--block_size",                     default=256, help="Optional input sequence length after tokenization")
parser.add_argument("--log_loss_interval",              default=25)
parser.add_argument("--device",                         default='cuda:0')
parser.add_argument("--num_train_epochs",               default=1)
parser.add_argument("--num_warmup_steps",               default=3000)
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


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path))
    model.resize_token_embeddings(len(tokenizer))

    train_new, test_new = ds.load_train_test_dataset(comments_path='data/UScomments.csv',
                                            videos_path='data/USvideos.csv',
                                            split=0.8,
                                            remove_extras=False,
                                            to_lower_case=False,
                                            remove_short_words=False,
                                            max_len=256)
    #print("Train:", train)
    #print("Test:", test)

    #encoded_train = train_old.map(tokenize_function, batched=True).shuffle(seed=42)
    #encoded_test = test_old.map(tokenize_function, batched=True).shuffle(seed=42)

    train_new.foreach(tokenize_function)
    test_new.foreach(tokenize_function)

    #train_dataset = encoded_train.foreach(group_texts, batched=True)
    #test_dataset = encoded_test.foreach(group_texts, batched=True)
    #train_new.foreach(group_texts)
    #test_new.foreach(group_texts)

    training_args = TrainingArguments(
        "gpt2-finetuned-ytcomments",
        learning_rate=5e-5,
        weight_decay=0.01,
        push_to_hub=False,
        evaluation_strategy="steps",
        logging_dir="logs_1",
        logging_steps=25,
        save_strategy='steps',
        save_steps = 10_000,
        load_best_model_at_end=True,
        eval_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_new,
        eval_dataset=test_new,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    model.save_pretrained('/Users/idorosh/hse/krylov/youtube_commenter-main/last_1')