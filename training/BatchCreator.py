import hashlib
from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch.functional import F

from tokenizer.TikTokenTokenizer import TikTokenTokenizer


class BatchCreator:
    @dataclass
    class DataDescriptor:
        data: list[str] | Iterator
        buffer: list[int] = field(default_factory=list)
        rows_consumed: int = 0


    def __init__(self, tokenizer: TikTokenTokenizer, device, val_size = 2000):
        self.pre_data = {}
        self.post_data = {}
        self.tokenizer = tokenizer
        self.device = device
        self.val_size = val_size

    def get_rows_consumed(self, phase, split):
        if phase == 'pre':
            return self.pre_data[split].rows_consumed
        elif phase == 'post':
            return self.post_data[split].rows_consumed
        else:
            raise ValueError(f"phase should be pre or post")

    def set_pre_data(self, data, skip_rows):
        def row_bucket(row, buckets=1000):
            key = row["id"]
            digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            return value % buckets

        def is_val_row(row):
            return row_bucket(row) == 0

        def is_train_row(row):
            return row_bucket(row) != 0

        train_dataset = data.filter(is_train_row).skip(skip_rows)
        val_dataset = list(data.filter(is_val_row).take(self.val_size))

        self.pre_data["train"] = BatchCreator.DataDescriptor(iter(train_dataset))
        self.pre_data["val"] = BatchCreator.DataDescriptor(val_dataset)

    def set_post_data(self, split, data, skip_rows):
        self.post_data[split] = BatchCreator.DataDescriptor(data)

    def reset_data(self, phase, split):
        if phase == 'pre':
            self.pre_data[split].buffer = []
            self.pre_data[split].rows_consumed = 0
        elif phase == 'post':
            self.post_data[split].buffer = []
            self.post_data[split].rows_consumed = 0
        else:
            raise ValueError(f"phase should be pre or post")

    def get_pre_batch(self, split, batch_size, block_size):
        assert split in self.pre_data.keys()

        data = self.pre_data[split].data
        buffer = self.pre_data[split].buffer
        needed_tokens = batch_size * block_size + 1
        rows_consumed = 0

        while len(buffer) < needed_tokens:
            if split == 'train':
                next_chunk = next(data)["text"] + '\n' + TikTokenTokenizer.eos_token_str
            else:
                next_chunk = data[self.pre_data[split].rows_consumed % len(data)]["text"]  + '\n' + TikTokenTokenizer.eos_token_str

            rows_consumed += 1
            enc_chunk = self.tokenizer.encode(next_chunk)
            buffer.extend(enc_chunk)

            self.pre_data[split].rows_consumed += 1

        chunks = buffer[:needed_tokens]
        chunk_tensor = torch.tensor(chunks, dtype=torch.long)

        x = torch.stack([chunk_tensor[batch:batch+block_size] for batch in range(0, batch_size * block_size, block_size)])
        y = torch.stack([chunk_tensor[batch + 1:batch+block_size+1] for batch in range(0, batch_size * block_size, block_size)])
        x = x.to(self.device, dtype=torch.long)
        y = y.to(self.device, dtype=torch.long)
        self.pre_data[split].buffer = buffer[needed_tokens - 1:]
        return x, y, rows_consumed

    def get_post_batch(self, split, batch_size, block_size):
        data = self.post_data[split].data

        if len(data) == 0:
            raise ValueError(f"No {split} post-training data loaded")

        system_token = self.tokenizer.encode(TikTokenTokenizer.system_token_str)[0]
        user_token = self.tokenizer.encode(TikTokenTokenizer.user_token_str)[0]
        assistant_token = self.tokenizer.encode(TikTokenTokenizer.assistant_token_str)[0]
        newline_tokens = set(self.tokenizer.encode("\n"))
        pad_token = self.tokenizer.get_eos_token()
        role_tokens = {system_token, user_token, assistant_token}

        xs = []
        ys = []
        attempts = 0
        max_attempts = batch_size * 20

        while len(xs) < batch_size and attempts < max_attempts:
            attempts += 1
            batch = torch.randint(len(data), (1,)).item()
            tokens = data[batch]

            # Bug fix at the moment to not train on chats that exceed the block_size
            if len(tokens) < 2 or len(tokens) > block_size + 1:
                continue

            y = tokens[1:].clone()
            assistant_response = False
            mask_next_newline = False

            # Mask output ys for non-assistant response
            for i in range(y.shape[0]):
                token = y[i].item()

                if token in role_tokens:
                    assistant_response = token == assistant_token
                    mask_next_newline = True
                    y[i] = -100
                    continue

                if mask_next_newline and token in newline_tokens:
                    mask_next_newline = False
                    y[i] = -100
                    continue

                mask_next_newline = False

                if not assistant_response:
                    y[i] = -100

            if len(tokens) > block_size + 1:
                start = len(tokens) - block_size - 1
                x = tokens[start:-1].clone()
                y = y[start:]
            else:
                x = tokens[:-1].clone()

            if torch.all(y == -100):
                continue

            xs.append(x)
            ys.append(y)

        if len(xs) < batch_size:
            raise ValueError("Could not build a post-training batch with valid assistant labels")

        pad_len = max(len(x) for x in xs)
        x_padded = [F.pad(x, (0, pad_len - len(x)), value=pad_token) for x in xs]
        y_padded = [F.pad(y, (0, pad_len - len(y)), value=-100) for y in ys]

        x = torch.stack(x_padded)
        y = torch.stack(y_padded)

        x = x.to(self.device, dtype=torch.long)
        y = y.to(self.device, dtype=torch.long)

        return x, y