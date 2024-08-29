from torch.utils.data import IterableDataset, DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
from datasets import load_dataset


class StreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, infinite=False, buffer_size=1000):
        super().__init__()
        self.dataset = dataset.filter(lambda example: example['label'] in [0, 1, 2])
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

train_dataset = load_dataset('nyu-mll/multi_nli', split='train', streaming=True)
train_dataset = train_dataset.shuffle(buffer_size=10000)

eval_dataset = load_dataset('nyu-mll/multi_nli', split='validation_matched', streaming=True)
eval_dataset = eval_dataset.shuffle(buffer_size=1000)

test_dataset = load_dataset('nyu-mll/multi_nli', split='validation_mismatched', streaming=True)
test_dataset = test_dataset.shuffle(buffer_size=1000)

train_dataset = StreamingDataset(train_dataset, tokenizer, infinite=True)
eval_dataset = StreamingDataset(eval_dataset, tokenizer, infinite=False)
test_dataset = StreamingDataset(test_dataset, tokenizer, infinite=False)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
