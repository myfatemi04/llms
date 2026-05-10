""" """

from functools import partial
import os

import transformers
import datasets
import time
import torch.utils.data
import torch


def encode(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )


def load_dataset(tokenizer):
    # By default, it's a DatasetDict with a 'train' split. Select the 'train' split to make it not a DatasetDict object.
    if not os.path.exists("scratch/alpaca_dataset_encoded"):
        dataset = datasets.load_dataset("tatsu-lab/alpaca", split="train")

        t0 = time.time()
        dataset = dataset.map(partial(encode, tokenizer=tokenizer), batched=True)
        t1 = time.time()

        print("Time taken to encode the dataset: ", t1 - t0, " seconds")

        # Save encoded dataset to disk
        dataset.save_to_disk("scratch/alpaca_dataset_encoded")
    else:
        dataset = datasets.load_from_disk("scratch/alpaca_dataset_encoded")

    # Split into 5% validation
    dataset = dataset.train_test_split(test_size=0.05)  # type: ignore

    train = dataset["train"]
    val = dataset["test"]

    return train, val


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)

    train, val = load_dataset(tokenizer)

    # print("Train sample:")
    # print(train[0])

    # print("Validation sample:")
    # print(val[0])

    # Create DataLoader.
    dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)

    for batch in dataloader:
        batch = {k: torch.tensor(v, device=device) for k, v in batch.items()}
        print(batch.keys())
        prediction = model(**batch)
        print(prediction.keys())
        break


if __name__ == "__main__":
    main()
