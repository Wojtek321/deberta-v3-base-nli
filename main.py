from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer
import torch
from data import data_collator, train_dataset, eval_dataset
from sklearn.metrics import accuracy_score, f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'microsoft/deberta-v3-base'

id2label = {
    '0': 'entailment',
    '1': 'neutral',
    '2': 'contradiction',
}

label2id = {v: k for k, v in id2label.items()}


config = AutoConfig.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)

def eval(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    return {"accuracy": accuracy, "f1": f1}



training_args = TrainingArguments(
    output_dir="results",
    max_steps=30000,
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=1e-3,
    evaluation_strategy="steps",
    eval_steps=6000,
    lr_scheduler_type="cosine",
    warmup_steps=500,

    logging_strategy="steps",
    logging_steps=200,

    save_steps=6000,
    save_only_model=True,

    push_to_hub=False,
    hub_model_id="deberta-v3-base-nli",
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=eval
)

trainer.train()
trainer.push_to_hub()