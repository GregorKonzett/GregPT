import pandas as pd
import matplotlib.pyplot as plt

file = "../data/progress.jsonl"

with open(file, "r") as f:
    progress = pd.read_json(f, lines=True)

training_lines = progress[progress["event"] == "train"]
validation_lines = progress[progress["event"] == "val"]

plt.plot(training_lines["global_step"], training_lines["train_loss"], label='Training loss', color='blue', marker='o')
plt.plot(validation_lines["global_step"], validation_lines["val_loss"], label='Validation loss', color='orange', marker='s')
plt.legend()

plt.show()