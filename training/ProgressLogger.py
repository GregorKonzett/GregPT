import json


class ProgressLogger:
    def __init__(self, file_path):
        self.file_path = file_path

    def log(self, data):
        print(data)
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(data) + "\n")