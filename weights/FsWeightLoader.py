import os

import torch

from weights.WeightLoader import WeightLoader


class FsWeightLoader(WeightLoader):
    def __init__(self):
        super().__init__()

    def load_checkpoint(self, gpt, optimizer=None):
        if not os.path.exists(self.model_weight_path):
            print("No checkpoint found")
            return 0

        checkpoint = torch.load(
            self.model_weight_path,
            weights_only=True,
            map_location=self.device,
        )

        return self.load(gpt, optimizer, checkpoint)

    def store_checkpoint(self, state_dict, global_step, optimizer, loss=None):
        self.store(state_dict, global_step, optimizer, loss)