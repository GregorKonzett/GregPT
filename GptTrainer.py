import torch
from torch import Tensor
from GptModel import GptModel, block_size, get_device
from weights.WeightLoader import WeightLoader

batch_size = 32
learning_rate = 3e-4
eval_iters = 200

class GptTrainer:
    def __init__(self, gpt: GptModel, weight_loader: WeightLoader):
        self.device = get_device()
        self.gpt = gpt.to(self.device)
        self.weight_loader = weight_loader
        self.training_data = torch.zeros((1,1), dtype=torch.long, device=self.device)
        self.eval_data = torch.zeros((1,1), dtype=torch.long, device=self.device)

    def get_batch(self, split: str = ''):
        data = self.training_data if split == 'train' else self.eval_data
        data_len = len(data)
        batches = torch.randint(data_len - block_size, (batch_size,))
        x = torch.stack([data[batch:batch+block_size] for batch in batches])
        y = torch.stack([data[batch + 1:batch+block_size+1] for batch in batches])
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.gpt.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self.gpt(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.gpt.train()
        return out

    def pre_train(self, iters, training_data: Tensor, eval_data: Tensor, load_checkpoint=False):
        print(f"Pre-training with {iters} iterations")
        self.training_data = training_data.to(self.device, dtype=torch.long)
        self.eval_data = eval_data.to(self.device, dtype=torch.long)
        self.__train(iters, load_checkpoint)

    def post_train(self, iters, load_checkpoint=False):
        pass

    def __train(self, iters, load_checkpoint=False):
        print("Training")

        optimizer = torch.optim.Adam(self.gpt.parameters(), lr=learning_rate)

        global_step = 0

        if load_checkpoint:
            global_step = self.weight_loader.load_checkpoint(self.gpt, optimizer)

        for iter in range(iters):
            xb, yb = self.get_batch("train")
            logits, loss = self.gpt(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 100 == 0:
                self.weight_loader.store_checkpoint(self.gpt.state_dict(), global_step, optimizer, loss)
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")