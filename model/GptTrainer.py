import torch
from torch.functional import F
from torch import Tensor
from model.GptModel import GptModel, block_size, get_device
from tokenizer.TikTokenTokenizer import TikTokenTokenizer
from weights.WeightLoader import WeightLoader

batch_size = 8
learning_rate = 3e-4
eval_iters = 200

class GptTrainer:
    def __init__(self, gpt: GptModel, weight_loader: WeightLoader, tokenizer: TikTokenTokenizer):
        self.device = get_device()
        self.tokenizer = tokenizer
        self.gpt = gpt.to(self.device)
        self.weight_loader = weight_loader
        self.training_data = None
        self.eval_data = None
        self.buffer = []

    def get_batch(self):
        data = self.training_data
        needed_tokens = batch_size * block_size + 1

        while len(self.buffer) < needed_tokens:
            next_chunk = next(data)["text"] + '\n' + TikTokenTokenizer.eos_token_str
            enc_chunk = self.tokenizer.encode(next_chunk)
            self.buffer.extend(enc_chunk)

        chunks = self.buffer[:needed_tokens]
        chunk_tensor = torch.tensor(chunks, dtype=torch.long)

        x = torch.stack([chunk_tensor[batch:batch+block_size] for batch in range(0, batch_size * block_size, block_size)])
        y = torch.stack([chunk_tensor[batch + 1:batch+block_size+1] for batch in range(0, batch_size * block_size, block_size)])
        x = x.to(self.device, dtype=torch.long)
        y = y.to(self.device, dtype=torch.long)
        self.buffer = self.buffer[needed_tokens - 1:]
        return x, y

    def get_post_batch(self, split: str = ''):
        data = self.training_data if split == 'train' else self.eval_data
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

            if len(tokens) < 2:
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

    @torch.no_grad()
    def estimate_loss(self, phase):
        out = {}
        self.gpt.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                if phase == 'pre':
                    X, Y = self.get_batch()
                elif phase == 'post':
                    X, Y = self.get_post_batch(split)
                else:
                    raise ValueError(f"phase should be pre or post")

                logits, loss = self.gpt(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.gpt.train()
        return out

    def pre_train(self, iters, training_data, load_checkpoint: bool = False):
        print(f"Pre-training with {iters} iterations")

        self.training_data = iter(training_data)

        self.__train("pre", iters, load_checkpoint)

    def post_train(self, iters, training_data: list[Tensor], eval_data: list[Tensor], load_checkpoint=False):
        print(f"Post-training with {iters} iterations")

        self.training_data = training_data
        self.eval_data = eval_data

        self.__train("post", iters, load_checkpoint)

    def __train(self, phase, iters, load_checkpoint: bool = False):
        optimizer = torch.optim.Adam(self.gpt.parameters(), lr=learning_rate)

        global_step = 0

        if load_checkpoint:
            global_step = self.weight_loader.load_checkpoint(self.gpt, optimizer)

        for iter in range(iters):
            if phase == 'pre':
                xb, yb = self.get_batch()
            elif phase == 'post':
                xb, yb = self.get_post_batch("train")
            else:
              raise ValueError("phase should be pre or post")

            logits, loss = self.gpt(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 100 == 0:
                self.weight_loader.store_checkpoint(self.gpt.state_dict(), global_step, optimizer, loss)

                if phase == 'pre':
                    print(f"step {iter}: train loss {loss.item():.4f}")
                else:
                    losses = self.estimate_loss(phase)
                    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
