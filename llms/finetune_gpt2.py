""" """

from functools import partial
import os

import transformers
import datasets
import time
import torch.utils.data
import torch
import wandb


def encode(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        # padding="max_length",
        padding=True,
        max_length=512,
        return_tensors="pt",
        return_length=True,
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
    # dataset = datasets.load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.train_test_split(test_size=0.05)  # type: ignore

    train = dataset["train"]
    val = dataset["test"]

    return train, val


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    train, val = load_dataset(tokenizer)

    wandb.init(project="finetune-gpt2", name="gpt2-medium-alpaca")

    # Create DataLoader.
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    t0 = time.time()

    for epoch in range(10):
        for batch in dataloader:
            # batch = {
            #     "input_ids": torch.stack(batch["input_ids"], dim=-1).to(device),
            #     "attention_mask": torch.stack(batch["attention_mask"], dim=-1).to(device),
            # }
            # print(batch["input_ids"].shape, batch["attention_mask"].shape)

            batch = tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_length=True,
            ).to(device)

            # First token's logit should target the second token
            logits = model(**batch)["logits"][:, :-1].contiguous()
            targets = batch["input_ids"][:, 1:].contiguous()

            # b1 = torch.arange(batch["input_ids"].size(0))
            # b2 = batch["length"] - 1
            # print(b1.shape, b2.shape)
            # targets[b1, b2:] = -100
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            wandb.log(
                {
                    "loss": loss.item(),
                    "elapsed_time": time.time() - t0,
                }
            )
            optim.zero_grad()
            loss.backward()
            optim.step()


if __name__ == "__main__":
    main()
