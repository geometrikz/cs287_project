from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import logging
import sys
import argparse
import os

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import nltk
import torch
import transformers
from datasets import load_dataset, load_metric
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from datasets import load_dataset


# Set up logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":

    logger.info(sys.argv)

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str, default='facebook/bart-large')
    parser.add_argument("--learning_rate", type=str, default=3e-5)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_source_length", type=str, default=1024)
    parser.add_argument("--fp16", type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    
    
    config = AutoConfig.from_pretrained(
    args.model_name, cache_dir=None, revision="main", use_auth_token=False,
    max_position_embeddings = args.max_source_length
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
    args.model_name, cache_dir=None, use_fast=True, revision="main", use_auth_token=False,
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
    args.model_name, from_tf=bool(".ckpt" in MODEL_NAME),
    config=None, cache_dir=None, revision="main",
    use_auth_token=False,)
    
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=label_pad_token_id)

    NUM_GPU = args.n_gpus
    check_val = False

    args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True if check_val == True else False,
        evaluation_strategy="steps",
        eval_steps=10000,
        logging_dir=f"{args.output_data_dir}/logs",
        # num_train_epochs=1,
        logging_steps=1000,
        
        max_steps=int(15000 * 8 / NUM_GPU),
#         num_train_epochs=args.epoch,
        
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=None,
        lr_scheduler_type='polynomial',
        learning_rate=float(args.learning_rate),
        warmup_steps=500,
        save_steps=20000,
        generation_max_length=64,
        
        fp16=args.fp16
    )


    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset= eval_dataset if check_val == True else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics if check_val == True else None
    )

    # train model
    if get_last_checkpoint(args.output_dir) is not None:
        logger.info("***** continue training *****")
        last_checkpoint = get_last_checkpoint(args.output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)