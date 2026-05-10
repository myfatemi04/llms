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


@torch.no_grad()
def validate(model, tokenizer, val_dataloader):
    # Generate up to 128 tokens for each prompt in the validation set.
    # Furthermore, compute perplexity of text. Samples with a temperature
    # of 0.7.

    result = {"val_loss": 0.0, "instruction": [], "completion": []}

    for batch in val_dataloader:
        #### Validation Loss ####
        tokenized = tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_length=True,
        ).to(model.device)

        # print("=== Text ===")
        # print(batch["text"][0])
        # print()
        # print("=== Instruction ===")
        # print(batch["instruction"][0])
        # print()
        # print("=== Input ===")
        # print(batch["input"][0])
        # print()
        # print("=== Output ===")
        # print(batch["output"][0])
        # print()

        logits = model(**tokenized)["logits"][:, :-1].contiguous()
        targets = tokenized["input_ids"][:, 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )

        result["val_loss"] += loss.item()

        #### Generation Quality ####
        model.eval()
        for instruction in batch["instruction"]:
            prompt_text = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:".format(
                instruction
            )
            prompt_input = tokenizer([prompt_text], return_tensors="pt")

            generated = model.generate(
                prompt_input["input_ids"].to(model.device),
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
            )
            # Decode generated tokens to text
            generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)

            result["instruction"].append(instruction)
            result["completion"].append(generated_text[0][len(prompt_text) :])

            # print("=== Prompt ===")
            # print(prompt_text)
            # print()

            # print("=== Generated Text ===")
            # print(generated_text[0])

            # print("=== Example Response ===")
            # print(batch["output"][0])
            # print()

        model.train()
        break

    result["val_loss"] /= len(result["completion"])

    return result


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
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    t0 = time.time()

    for epoch in range(10):
        # Run validation step. Do before training loop so we see what the results looked like
        # before we did any fine-tuning.
        val_result = validate(model, tokenizer, val_dataloader)

        wandb.log(
            {"val_loss": val_result["val_loss"] / len(val_dataloader), "epoch": epoch}
        )

        # Create table
        table = wandb.Table(columns=["instruction", "completion"])
        for instruction, completion in zip(
            val_result["instruction"], val_result["completion"]
        ):
            table.add_data(instruction, completion)

        wandb.log({"validation_generations": table, "epoch": epoch})

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
                    "epoch": epoch,
                }
            )
            optim.zero_grad()
            loss.backward()
            optim.step()


if __name__ == "__main__":
    main()
