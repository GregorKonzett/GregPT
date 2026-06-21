import math

import torch
from torch import Tensor
from model.GptModel import GptModel, block_size, get_device
from tokenizer.TikTokenTokenizer import TikTokenTokenizer
from training.ProgressLogger import ProgressLogger
from training.BatchCreator import BatchCreator
from weights.WeightLoader import WeightLoader

batch_size = 8
gradient_accumulation_steps = 4
learning_rate = 3e-4
min_learning_rate = 3e-5
eval_iters = 50
betas = (0.9, 0.95)
weight_decay = (0.1, 0.0)
iters_between_val = 1000
iters_between_log = 100
max_decay_steps = 10_000


def lr_lambda(step):
    if step >= max_decay_steps:
        return min_learning_rate / learning_rate

    progress = step / max_decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = min_learning_rate + cosine * (learning_rate - min_learning_rate)

    return lr / learning_rate

class GptTrainer:
    def __init__(self, gpt: GptModel, weight_loader: WeightLoader, tokenizer: TikTokenTokenizer):
        self.device = get_device()
        self.tokenizer = tokenizer
        self.gpt = gpt.to(self.device)
        self.weight_loader = weight_loader
        self.batch_creator = BatchCreator(tokenizer, self.device)
        self.progress_loader = ProgressLogger("./data/progress.jsonl")

    def find_decay_groups(self):
        decay = []
        no_decay = []

        for name, param in self.gpt.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim >= 2 and "token_embedding_table" not in name:
                decay.append(param)
            else:
                no_decay.append(param)

        return decay, no_decay

    @torch.no_grad()
    def estimate_val_loss(self, phase):
        self.gpt.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if phase == 'pre':
                X, Y, _ = self.batch_creator.get_pre_batch("val", batch_size, block_size)
            elif phase == 'post':
                X, Y = self.batch_creator.get_post_batch("val", batch_size, block_size)
            else:
                raise ValueError(f"phase should be pre or post")

            logits, loss, _ = self.gpt(X, Y)
            losses[k] = loss.item()
        self.gpt.train()

        return losses.mean()

    def pre_train(self, iters, training_data, load_checkpoint: bool = False):
        print(f"Pre-training with {iters} iterations")

        self.__train("pre", training_data, iters, load_checkpoint)

    def post_train(self, iters, training_data: list[Tensor], eval_data: list[Tensor], load_checkpoint=False):
        print(f"Post-training with {iters} iterations")

        self.batch_creator.set_post_data("val", eval_data, 0)

        self.__train("post", training_data, iters, load_checkpoint)

    def __train(self, phase, training_data, iters, load_checkpoint: bool = False):
        decay, no_decay = self.find_decay_groups()

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay[0]},
                {"params": no_decay, "weight_decay": weight_decay[1]},
            ],
            lr=learning_rate,
            betas=(0.9, 0.95),
        )

        global_step = 0

        # Add optimizer learning rate decay that clamps at min_learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

        if load_checkpoint:
            global_step, rows_consumed, tokens_seen = self.weight_loader.load_checkpoint(self.gpt, optimizer, scheduler)
        else:
            rows_consumed = 0
            tokens_seen = 0

        if phase == 'pre':
            self.batch_creator.set_pre_data(training_data, rows_consumed)
        elif phase == 'post':
            self.batch_creator.set_post_data("train", training_data, 0)

        train_losses = torch.zeros(iters_between_val)

        for i in range(iters):
            optimizer.zero_grad()
            step_loss = 0.0

            for micro_step in range(gradient_accumulation_steps):
                if phase == 'pre':
                    xb, yb, rows_consumed_tmp = self.batch_creator.get_pre_batch("train", batch_size, block_size)
                elif phase == 'post':
                    xb, yb = self.batch_creator.get_post_batch("train", batch_size, block_size)
                    rows_consumed_tmp = 0
                else:
                  raise ValueError("phase should be pre or post")

                rows_consumed += rows_consumed_tmp

                logits, loss, _ = self.gpt(xb, yb)

                step_loss += loss.item()
                loss /= gradient_accumulation_steps
                loss.backward()

            tokens_seen += gradient_accumulation_steps * batch_size * block_size

            torch.nn.utils.clip_grad_norm_(self.gpt.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            train_losses[i % iters_between_val] = step_loss / gradient_accumulation_steps

            if global_step % iters_between_log == 0:
                self.progress_loader.log({
                    "global_step": global_step,
                    "event": "train",
                    "tokens_seen": tokens_seen,
                    "train_loss": step_loss / gradient_accumulation_steps,
                    "rows_consumed": rows_consumed,
                    "lr": optimizer.param_groups[0]['lr'],
                })
                # print(f"{global_step}: {tokens_seen}\t{step_loss / gradient_accumulation_steps:.4f}\t{optimizer.param_groups[0]['lr']}")

            if global_step % iters_between_val == 0:
                train_loss = train_losses.mean()
                val_los = self.estimate_val_loss(phase)

                self.progress_loader.log({
                    "global_step": global_step,
                    "event": "val",
                    "tokens_seen": tokens_seen,
                    "train_loss": train_loss.item(),
                    "val_loss": val_los.item(),
                    "rows_consumed": rows_consumed,
                    "lr": optimizer.param_groups[0]['lr'],
                })

                self.weight_loader.store_checkpoint(
                    self.gpt.state_dict(),
                    global_step,
                    optimizer,
                    scheduler,
                    rows_consumed,
                    tokens_seen,
                    optimizer.param_groups[0]['lr'],
                    train_loss,
                    val_los
                )

                train_losses = torch.zeros(iters_between_val)

                # print(f"{global_step}: train loss {train_loss:.4f}, val loss {val_los:.4f}")