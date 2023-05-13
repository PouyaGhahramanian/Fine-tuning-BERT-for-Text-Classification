import huggingface_hub
from huggingface_hub import notebook_login
from datasets import load_dataset, load_metric
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, BertTokenizer, BertModel, AutoConfig, BertConfig
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, BertForSequenceClassification
import torch
from transformers import logging
import optuna
import torch.nn as nn
import sqlite3
import warnings
warnings.filterwarnings("ignore")

# Using SQLite to save the optuna results
conn = sqlite3.connect('optuna_p1.db')

dataset = load_dataset("glue", "cola")
train_dataset, val_dataset = dataset["train"], dataset["validation"]

# Helper methods to tokenize the dataset and convert to Pytorch format
def tokenize(batch, max_seq_length):
    return tokenizer(batch["sentence"], truncation=True, padding=True, max_length=max_seq_length)

def tokenize_dataset(dataset, max_seq_length):
    return dataset.map(lambda batch: tokenize(batch, max_seq_length), batched=True, batch_size=len(dataset))

# Set datasets format to PyTorch
def set_format(dataset):
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return dataset

torch.cuda.set_device(1)

logging.set_verbosity_error()

# Login to HuggingFace to push model
access_token = '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'
huggingface_hub.login(token = access_token)

task = "cola"
model_checkpoint = "bert-base-uncased"
batch_size = 64

dataset = load_dataset("glue", task)
metric = load_metric('glue', task)

max_seq_length = 128

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, truncation=True, padding=True, max_length=max_seq_length)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

def preprocess_function(examples, max_seq_length):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding=True, max_length=max_seq_length)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True, max_length=max_seq_length)

encoded_dataset = dataset.map(lambda batch: preprocess_function(batch, max_seq_length), batched=True)
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

model_checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
model.dropout.p = 0.2

metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

# For fine-tuning without hyperparameter tuning, uncomment the following lines
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset[validation_key],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# trainer.train()
# trainer.evaluate()
# trainer.push_to_hub()

# Hyperparameter optimization with optuna
def model_init(dp = 0.2):
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    # config_with_p = BertConfig.from_pretrained(model_checkpoint, num_labels=2, hidden_dropout_prob=dp)
    # model_init_with_p = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config_with_p)
    # return model_init_with_p

def print_trial_params(trial):
    print(f"Trial {trial.number}: {trial.params}")

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    num_epochs = trial.suggest_int('num_epochs', 1, 10)
    max_seq_length = trial.suggest_int('max_seq_length', 32, 128)
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)

    train_data = set_format(tokenize_dataset(train_dataset, max_seq_length))
    val_data = set_format(tokenize_dataset(val_dataset, max_seq_length))

    args = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        evaluation_strategy = "epoch",
        # save_strategy = "epoch",
        save_strategy = "no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        # load_best_model_at_end=True,
        load_best_model_at_end=False,
        metric_for_best_model=metric_name,
        # dropout = dropout,
        push_to_hub=True,
        save_total_limit = 1,
    )
    trainer = Trainer(
        # model_init=model_init(dropout),
        model_init=model_init,
        args=args,
        # train_dataset=encoded_dataset["train"],
        # eval_dataset=encoded_dataset[validation_key],
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_output = trainer.evaluate()
    mcc = eval_output['eval_matthews_correlation']
    loss = eval_output['eval_loss']

    trial.set_user_attr("intermediate_loss", loss)
    trial.set_user_attr("intermediate_mcc", mcc)

    print(f"Trial {trial.number}: {trial.params}")
    print('MCC: ', mcc)
    print('Loss: ', loss)

    return mcc

# Define the Optuna study
study = optuna.create_study(direction='maximize', storage='sqlite:///optuna_p2.db')
study.optimize(objective, n_trials=10)

best_params = study.best_params
print("Best hyperparameters:", best_params)

optuna.visualization.plot_optimization_history(study).write_image('p1_optuna_optimization_history.png')

trials = study.get_trials()
losses = [trial.user_attrs["intermediate_loss"] for trial in trials]
mcc_scores = [trial.user_attrs["intermediate_mcc"] for trial in trials]
epochs = list(range(1, len(trials) + 1))

print('Loss values: ', losses)
print('MCC values: ', mcc_scores)

plt.figure()
plt.plot(epochs, losses, label="Loss")
plt.xlabel("Trials")
plt.ylabel("Loss")
plt.legend()
plt.savefig("p1_loss_over_time.png")

plt.figure()
plt.plot(epochs, mcc_scores, label="MCC Score")
plt.xlabel("Trials")
plt.ylabel("MCC Score")
plt.legend()
plt.savefig("p1_mcc_over_time.png")

conn.close()

# trainer.push_to_hub()
