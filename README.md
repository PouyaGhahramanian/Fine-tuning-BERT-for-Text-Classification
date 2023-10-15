# Fine-tuning BERT for Text Classification
## Bilkent EEE586 Assignment II
### Assignment Description
For this assignment, you are required to work with BERT or one of its variants for the text classification task. Since the hardware of the students may vary (GPU sizes etc.) you are free to use the BERT variant of your choice which meets your hardware requirements. You will fine-tune a pre-trained model and explore several hyperparameters and their effect on the model’s performance.
### Abstract
In this assignment, we explore the Corpus of Linguistic Acceptability (CoLA)
dataset, one of the GLUE Benchmark datasets, for text classification us-
ing BERT-based models. First, we fine-tune a pretrained BERT model
(bert-base-uncased) on the GLUE dataset and tune learning rate, number of
epochs, max length of input sequence, and dropout hyperparameters using
the Optuna library. We use the loss and the Matthew Correlation Coeffi-
cient (MCC) as the evaluation metrics. We push our best-performing model
to the Hugging Face Hub. Then, we explore an alternative approach to
using [’CLS’] tokens as document representations. Similar hyperparameter
investigations are conducted.
### Introduction
Text classification is a fundamental task in natural language processing with
applications ranging from sentiment analysis to document categorization.
With the advent of pretrained language models like BERT (Bidirectional
Encoder Representations from Transformers), significant advancements have
been made in achieving state-of-the-art performance on various NLP bench-
marks. In this assignment, we delve into the Corpus of Linguistic Acceptabil-
ity (CoLA) dataset, a part of the General Language Understanding Evalua-
tion (GLUE) benchmark, to explore the effectiveness of BERT-based models
for text classification.
To begin our exploration, we fine-tune a pretrained BERT model, specifically
the "bert-base-uncased" variant, on the GLUE dataset. This process involves
adapting the model to the specific characteristics and requirements of the
CoLA dataset through a process known as fine-tuning.
In order to optimize the performance of our fine-tuned BERT model, we
tune several hyperparameters using the Optuna library. These hyperparam-
eters include the learning rate, number of epochs, maximum length of input
sequences, and dropout at the classification head. Optuna provides an effi-
cient framework for hyperparameter optimization, enabling us to search the
hyperparameter space and identify the optimal configuration that yields the
best performance.
To evaluate the performance of our models, we utilize both the loss function
and the Matthew Correlation Coefficient (MCC) as evaluation metrics. The
loss function provides an indication of the model’s training progress and
convergence, while the MCC takes into account both true positives and true
negatives, making it a suitable metric for imbalanced datasets like CoLA.
To ensure the reproducibility and accessibility of our best-performing model,
we leverage the Hugging Face Model Hub. By pushing our model to the
Hugging Face Hub, we make it readily available for others in the research
and NLP community to use and build upon, fostering collaboration and
enabling future advancements in text classification.
Additionally, we explore an alternative approach to utilizing BERT models
for text classification. Instead of solely relying on the "[CLS]" tokens as
document representations, we investigate an alternative technique by using
output of the all hidden states in the BERT model for document represen-
tations. We conduct a similar hyperparameter investigation, examining the
effects of learning rate, maximum input sequence length, and number of
epochs..
In the next section, we describe our architectures in details. We present our
results in section 4, and conclude the report in section 5.
### Architectures
For the first part I used the ’bert-base-uncased’ model from HuggingFace
and fine-tuned it on the GLUE dataset.
In the second part of the assignment we are required to use an alternative ap-
proach instead of using the [’CLS’] tokens for document representations. In
this part, I use an alternative approach by taking the average of the hidden
states of all tokens in the input sequence. Then, I use the average embed-
dings to a classification layer for final prediction. In other words, instead of
solely relying on the [CLS] token representation for the classification task,
we compute the average of the hidden states of all tokens in the input se-
quence to create a fixed-size document representation. This representation
is then passed through a linear classifier to produce the final output. To this
aim, I wrote a python class and named it as MeanPoolingBert that extends
BertForSequenceClassification model. I used the ’bert-base-uncased’ model
as the base model and modified the forward method to use the mean vector
of the hidden states as the embedding vector.
The intuition behind this approach is that by averaging the hidden states
of all tokens, the model can capture more meaningful information from the
entire input sequence, which can potentially lead to better classification per-
formance.
### Results
Available in the Report.pdf file.
### Discussions and Conclusions
Available in the Report.pdf file.
