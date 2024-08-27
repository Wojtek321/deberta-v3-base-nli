from torch.utils.data import IterableDataset, DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
from datasets import load_dataset


dataset = load_dataset('nyu-mll/multi_nli', split='train', streaming=True)
dataset = dataset.shuffle(buffer_size=10000)


class StreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, infinite=False, buffer_size=1000):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.infinite = infinite
        self.buffer_size = buffer_size

    def __iter__(self):
        iterator = iter(self.dataset)
        buffer = []

        while True:
            try:
                row = next(iterator)
                premise = row['premise']
                hypothesis = row['hypothesis']
                label = row['label']
                buffer.append((premise, hypothesis, label))

                if len(buffer) == self.buffer_size:
                    premises, hypotheses, labels = zip(*buffer)

                    inputs = self.tokenizer(list(premises), list(hypotheses), truncation=True, return_token_type_ids=True)
                    inputs['labels'] = labels

                    for i in range(self.buffer_size):
                        yield {key: value[i] for key, value in inputs.items()}

                    buffer = []

            except StopIteration:
                if self.infinite:
                    iterator = iter(self.dataset)
                else:
                    break


tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

prepared_train_dataset = StreamingDataset(dataset, tokenizer, infinite=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(prepared_train_dataset, batch_size=32, collate_fn=data_collator, pin_memory=True)
