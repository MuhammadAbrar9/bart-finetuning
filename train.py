"""
Creator & Developer : Muhammad Abrar
email : muhammadabrar9999@gmail.com
"""

import pdb
import logging
import sys
import math
import os
import wandb
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import datasets
from datasets import load_dataset, DatasetDict
from evaluate import load
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments

class PreprocessBart:

    def __init__(self, **kwargs):
        self.prefix = kwargs.get('prefix')
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs.get('model_checkpoint'))
        self.max_input_length = kwargs.get('max_input_length')
        self.max_target_length = kwargs.get('max_target_length')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(kwargs.get('model_checkpoint'))

    def preprocess_function(self, examples):
        inputs = [self.prefix + doc for doc in examples["sentence1"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        labels = self.tokenizer(text_target=examples["sentence2"], max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenized_data(self, datasets):
        tokenized_datasets = datasets.map(self.preprocess_function, batched=True)
        return tokenized_datasets

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer


def main():

    ts = datetime.datetime.now()
    print(f"\nCurrent Timestamp: {ts}\n")

    wandb.login(relogin=True, key="")
    wandb.init(project="bart_paraphrasing", name=f"retrain_{ts}") # entity="entity-x" 

    data_paths = "dataset-80-20/"
    data_files = {"train": "train-80.csv", "validation": "test-20.csv"}
    loaded_dataset = load_dataset(data_paths, data_files=data_files) # loading all the data
    
    # train_dataset = load_dataset(data_paths, data_files=data_files, split="train[:1%]")
    # validation_dataset = load_dataset(data_paths, data_files=data_files, split="validation[:1%]")
    # loaded_dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})

    max_input_length = 1024
    max_target_length = 128
    prefix = "paraphrase: "
    model_checkpoint = "stanford-oval/paraphraser-bart-large"
    bart = PreprocessBart(prefix=prefix, model_checkpoint=model_checkpoint,
                        max_input_length=max_input_length, max_target_length=max_target_length)
    tokenized_data = bart.tokenized_data(datasets=loaded_dataset)
    bart_model = bart.get_model()
    bart_tokenizer = bart.get_tokenizer()

    batch_size = 10
    model_name = model_checkpoint.split("/")[-1]
    ouptut_path = "model/"
    args = Seq2SeqTrainingArguments(
        # f"{model_name}-finetuned-chatgptphrases",
        output_dir = ouptut_path,
        overwrite_output_dir =True,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        no_cuda=False,
        ddp_backend='nccl',
        load_best_model_at_end=True
    )

    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # set the main code and the modules it uses to the same log-level according to the node
    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    data_collator = DataCollatorForSeq2Seq(bart_tokenizer, model=bart_model)

    trainer = Seq2SeqTrainer(
        bart_model,
        args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        data_collator=data_collator,
        tokenizer=bart_tokenizer,
        )

    print("Training Started: ")
    trainer.train()

    print("Evaluating: ")
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    print("\nSaving Model: ")
    model_loc = "/saved-model/"
    trainer.save_model(model_loc)



if __name__ == "__main__":
    main()



