from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer
import torch
from data import StreamingDataset, data_collator, prepared_train_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'microsoft/deberta-v3-small'

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



batch_size = 2


training_args = TrainingArguments(
    output_dir="model_checkpoints",
    max_steps=16,
    learning_rate=6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=1e-3,
    evaluation_strategy="steps",
    eval_steps=6000,
    lr_scheduler_type="cosine",
    warmup_steps=500,

    logging_strategy="steps",
    logging_steps=200,
    logging_first_step=True,

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
    train_dataset=prepared_train_dataset,
)

trainer.train()
