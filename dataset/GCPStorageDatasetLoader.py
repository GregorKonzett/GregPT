from google.cloud import storage

from dataset.DatasetLoader import DatasetLoader
from tokenizer import Tokenizer


class GCPStorageDatasetLoader(DatasetLoader):
    def __init__(self, path, name = None):
        super().__init__(path, name)

        self.file_blob = self.name if self.name else self.path.split("/")[-1]
        self.train_blob = self.file_blob + "train"
        self.val_blob = self.file_blob + "val"

        self.storage_client = storage.Client()
        self.bucket_name = "gregpt-datasets"
        self.bucket = self.storage_client.bucket(self.bucket_name)

        if not self.bucket.exists():
            self.bucket = self.storage_client.create_bucket(self.bucket_name)

    def __get_stored_data(self, split) -> str:
        if split == "train":
            blob = self.bucket.blob(self.train_blob)
        else:
            blob = self.bucket.blob(self.val_blob)

        print(f"Downloading {split} data from {blob}")

        return blob.download_as_text()

    def __write_data(self, split, text):
        if split == "train":
            blob = self.bucket.blob(self.train_blob)
        else:
            blob = self.bucket.blob(self.val_blob)

        print(f"Uploading {split} data to {blob}")

        blob.upload_from_string(text)

    def __blob_exists(self, split) -> bool:
        if split == "train":
            blob = self.bucket.blob(self.train_blob)
        else:
            blob = self.bucket.blob(self.val_blob)

        return blob.exists()

    def __get_data(self, phase, split) -> str:
        if self.__blob_exists(split):
            return self.__get_stored_data(split)

        text = self.download_data(phase, split)

        self.__write_data(split, text)
        return text

    def get_train_data(self, phase) -> str:
        return self.__get_data(phase, "train")

    def get_val_data(self, phase, split) -> str:
        return self.__get_data(phase, split)
