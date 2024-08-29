from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from data import data_collator, test_dataset
from sklearn.metrics import accuracy_score, f1_score
import torch
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "deberta-v3-base-nli"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    return {"accuracy": accuracy, "f1": f1}


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)


results = trainer.evaluate()

with open('result.json', 'w') as f:
    json.dump(results, f, indent=4)


model.save_pretrained("deberta-v3-base-nli", push_to_hub=True)
tokenizer.save_pretrained("deberta-v3-base-nli", push_to_hub=True)
