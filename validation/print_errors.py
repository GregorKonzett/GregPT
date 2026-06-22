import pandas as pd
import matplotlib.pyplot as plt

file = "../data/progress.jsonl"

with open(file, "r") as f:
    progress = pd.read_json(f, lines=True)

training_lines = progress[progress["event"] == "train"]
validation_lines = progress[progress["event"] == "val"]

# plt.legend()
#
# plt.show()


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('global steps')
ax1.set_ylabel('learning rate', color=color)
ax1.plot(training_lines["global_step"], training_lines["lr"], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('cross-entropy loss', color=color)
plt.plot(training_lines["global_step"], training_lines["train_loss"], label='Training loss', color='blue', marker='o')
plt.plot(validation_lines["global_step"], validation_lines["val_loss"], label='Validation loss', color='orange', marker='s')
ax2.tick_params(axis='y', labelcolor=color)

fig.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()