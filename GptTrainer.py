import torch
from torch import Tensor
import os
from GptModel import GptModel, block_size

batch_size = 32
learning_rate = 3e-4
model_weight_path = "./weights.pt"

class GptTrainer:
    def __init__(self, gpt: GptModel, data: Tensor, eval_data: Tensor, device):
        self.device = device
        self.gpt = gpt.to(self.device)
        self.training_data = data.to(self.device, dtype=torch.long)
        self.eval_data = eval_data.to(self.device, dtype=torch.long)

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

    def restore_checkpoint(self, optimizer):
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

    def train(self, iters, load_checkpoint=False):
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
                print("Iter:", iter, "Loss:", loss)