import os
import tempfile

import torch
from google.cloud import storage

from weights.WeightLoader import WeightLoader

class GCPStorageWeightLoader(WeightLoader):
    def __init__(self):
        super().__init__()

        self.storage_client = storage.Client()
        self.bucket_name = "gregpt-weights"
        self.blob_name = "weights.pt"
        self.tmp_file = os.path.join(tempfile.gettempdir(), "gregpt_weights.pt")
        self.model_weight_path = self.tmp_file
        self.bucket = self.storage_client.bucket(self.bucket_name)

        if not self.bucket.exists():
            self.bucket = self.storage_client.create_bucket(self.bucket_name)

    def __blob_exists(self) -> bool:
        blob = self.bucket.blob(self.blob_name)
        return blob.exists()

    def __load_blob(self):
        blob = self.bucket.blob(self.blob_name)
        blob.download_to_filename(self.tmp_file)

    def load_checkpoint(self, gpt, optimizer=None):
        if not self.__blob_exists():
            print("No checkpoint found")
            return 0

        print(f"Loading weights from checkpoint {self.bucket_name}/{self.blob_name}")
        self.__load_blob()

        try:
            checkpoint = torch.load(
                self.tmp_file,
                weights_only=True,
                map_location=self.device,
            )
        finally:
            if os.path.exists(self.tmp_file):
                os.remove(self.tmp_file)

        return self.load(gpt, optimizer, checkpoint)

    def store_checkpoint(self, state_dict, global_step, optimizer, loss=None):
        try:
            self.store(state_dict, global_step, optimizer, loss)
            blob = self.bucket.blob(self.blob_name)
            blob.upload_from_filename(self.tmp_file)
        finally:
            if os.path.exists(self.tmp_file):
                os.remove(self.tmp_file)
