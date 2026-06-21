import torch

from model.GptModel import GptModel, get_device


class WeightLoader:
    def __init__(self):
        self.device = get_device()
        self.model_weight_path = "./data/weights.pt"

    def load_checkpoint(self, gpt: GptModel, optimizer=None, scheduler=None):
        pass

    def store_checkpoint(self, state_dict, global_step, optimizer, scheduler, rows_consumed, tokens_seen, lr, train_loss=None, val_loss=None):
        pass

    def store(self, state_dict, global_step, optimizer, scheduler, rows_consumed, tokens_seen, lr, train_loss=None, val_loss=None):
        checkpoint = {
            "global_step": global_step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "lr": lr,
            "rows_consumed": rows_consumed,
            "tokens_seen": tokens_seen,
            "scheduler_state_dict": scheduler.state_dict(),
        }

        if torch.cuda.is_available():
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        if self.device.type == "mps":
            checkpoint["mps_rng_state"] = torch.mps.get_rng_state()

        if train_loss is not None:
            checkpoint["train_loss"] = train_loss.item()

        if val_loss is not None:
            checkpoint["val_loss"] = val_loss.item()

        torch.save(checkpoint, self.model_weight_path)

        print("Checkpoint saved")

    def load(self, gpt, optimizer, scheduler, checkpoint):
        gpt.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].cpu())

        if self.device.type == "mps" and "mps_rng_state" in checkpoint:
            torch.mps.set_rng_state(checkpoint["mps_rng_state"].cpu())

        if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all([
                state.cpu() for state in checkpoint["cuda_rng_state_all"]
            ])

        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        rows_consumed = checkpoint["rows_consumed"]
        tokens_seen = checkpoint["tokens_seen"]

        gpt.train()

        print("Checkpoint restored")

        return checkpoint.get("global_step", 0), rows_consumed, tokens_seen