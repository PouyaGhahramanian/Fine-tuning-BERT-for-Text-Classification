import huggingface_hub
from huggingface_hub import notebook_login
from datasets import load_dataset, load_metric
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, BertTokenizer, BertModel
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, BertForSequenceClassification
import torch
from transformers import logging

logging.set_verbosity_error()


task = "cola"
model_checkpoint = "bert-base-uncased"
batch_size = 16

dataset = load_dataset("glue", task)
metric = load_metric('glue', task)

# fake_preds = np.random.randint(0, 2, size=(64,))
# fake_labels = np.random.randint(0, 2, size=(64,))
# res = metric.compute(predictions=fake_preds, references=fake_labels)
# print(res)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# print(tokenizer("Hello, this one sentence!"))

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

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

# print(preprocess_function(dataset['train'][:5]))
encoded_dataset = dataset.map(preprocess_function, batched=True)

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

model_checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# model_name = 'bert-base-uncased'
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=False)

model.dropout.p = 0.2

metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]
# print(model_name)
# print(f"{model_name}-finetuned-{task}")
# pouya()

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)

# args = TrainingArguments(
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model=metric_name,
#     push_to_hub=True,
# )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

# Login to HuggingFace to push model
access_token = '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'
huggingface_hub.login(token = access_token)

trainer.push_to_hub()
