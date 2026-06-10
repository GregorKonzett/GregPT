from GptModel import GptModel
from GptTrainer import GptTrainer
import torch
from tokenizer import Tokenizer

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

tokenizer = Tokenizer()

vocab_size = tokenizer.get_vocab_size()

encoded_text = torch.tensor(tokenizer.encode(text), dtype=torch.long)
training_length = int(len(encoded_text) * 0.9)
eval_length = len(encoded_text) - training_length

training_data = encoded_text[:training_length]
eval_data = encoded_text[training_length:]

gpt = GptModel(vocab_size=vocab_size, device=device)
params = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
print(f"Parameters: {params}")
trainer = GptTrainer(gpt, training_data, eval_data, device)

iters = 3000

trainer.train(iters, True)
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
out = gpt.generate(idx, 256)
print("Output:")
print(tokenizer.decode(out[0].cpu().tolist()))