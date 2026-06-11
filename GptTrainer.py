import torch
from torch import Tensor
import os
from GptModel import GptModel, block_size, get_device

batch_size = 32
learning_rate = 3e-4
eval_iters = 200
model_weight_path = "./weights.pt"

class GptTrainer:
    def __init__(self, gpt: GptModel):
        self.device = get_device()
        self.gpt = gpt.to(self.device)
        self.training_data = torch.zeros((1,1), dtype=torch.long, device=self.device)
        self.eval_data = torch.zeros((1,1), dtype=torch.long, device=self.device)

    def get_batch(self, split: str = ''):
        data = self.training_data if split == 'train' else self.eval_data
        data_len = len(data)
        batches = torch.randint(data_len - block_size, (batch_size,))
        x = torch.stack([data[batch:batch+block_size] for batch in batches])
        y = torch.stack([data[batch + 1:batch+block_size+1] for batch in batches])
        return x, y

    def save_checkpoint(self, global_step, optimizer, loss=None):
        checkpoint = {
            "global_step": global_step,
            "model_state_dict": self.gpt.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        if self.device.type == "mps":
            checkpoint["mps_rng_state"] = torch.mps.get_rng_state()

        if loss is not None:
            checkpoint["loss"] = loss.item()

        torch.save(checkpoint, model_weight_path)

        print("Checkpoint saved")

    def restore_checkpoint(self, optimizer=None):
        if not os.path.exists(model_weight_path):
            print("No checkpoint found")
            return 0

        checkpoint = torch.load(
            model_weight_path,
            weights_only=True,
            map_location=self.device,
        )

        self.gpt.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].cpu())

        if self.device.type == "mps" and "mps_rng_state" in checkpoint:
            torch.mps.set_rng_state(checkpoint["mps_rng_state"].cpu())

        if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

        self.gpt.train()

        print("Checkpoint restored")

        return checkpoint.get("global_step", 0)

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
            global_step = self.restore_checkpoint(optimizer)

        for iter in range(iters):
            xb, yb = self.get_batch("train")
            logits, loss = self.gpt(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 100 == 0:
                self.save_checkpoint(global_step, optimizer, loss)
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")