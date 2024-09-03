# deberta v3 base - Natural Language Inference

## Model overview
This repository contains the script used to finetune [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) model using the [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) dataset. Finetuned model can be found [here](https://huggingface.co/chincyk/deberta-v3-base-nli).
This model is trained for the Natural Language Inference task. It takes two sentences as input (a premise and a hypothesis) and predicts the relationship between them by assigning one of three labels: "entailment," "neutral," or "contradiction" and returns scores corresponding to the labels.

## Results
After fine-tuning on the dataset, the model achieved the following results:

- Loss: 0.276
- Accuracy: 0.899
- F1-Score: 0.899

These metrics were evaluated on the `validation_mismatched` split of the dataset.

## Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "chincyk/deberta-v3-base-nli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

premise = "The flight arrived on time at the airport."
hypothesis = "The flight was delayed by several hours."
inputs = tokenizer(premise, hypothesis, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
probs = torch.softmax(logits, dim=-1).squeeze()

id2label = model.config.id2label

for i, prob in enumerate(probs):
    print(f"{id2label[i]}: {prob:.4f}")
```
