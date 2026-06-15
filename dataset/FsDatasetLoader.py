import os

from datasets import load_dataset

from dataset.DatasetLoader import DatasetLoader

class FsDatasetLoader(DatasetLoader):
    def __init__(self, path, name = None):
        super().__init__(path, name)

        self.file_path = "./data/" + self.name
        self.train_path = self.file_path + "train.txt"
        self.val_path = self.file_path + "val.txt"

    def get_data(self, phase: str, split: str) -> str:
        if split == "train":
            file_path = self.train_path
        else:
            file_path = self.val_path

        if os.path.exists(file_path):
            print("Found data locally")
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        text = self.download_data(phase, split)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
            print("Saved data to file")

        return text

    def get_train_data(self, phase) -> str:
        return self.get_data(phase, "train")

    def get_val_data(self, phase, split) -> str:
        return self.get_data(phase, split)