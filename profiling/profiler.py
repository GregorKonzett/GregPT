import torch

from model.GptModel import GptModel, get_device
from tokenizer.TikTokenTokenizer import TikTokenTokenizer
from weights.GCPStorageWeightLoader import GCPStorageWeightLoader

trace_path = "../data/traces/trace"
tokenizer = TikTokenTokenizer()
weight_loader = GCPStorageWeightLoader()
gpt = GptModel(vocab_size=tokenizer.get_vocab_size(), tokenizer=tokenizer)
gpt = gpt.to(get_device())
pytorch_total_params = sum(p.numel() for p in gpt.parameters())

query = "What is 2+2"
weight_loader.load_checkpoint(gpt, False)
encoded_query = torch.tensor([tokenizer.encode(query)], dtype=torch.long, device=get_device())

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,  # the cpu activities
        torch.profiler.ProfilerActivity.CUDA, # the gpu activities
    ],
) as prof:
    for _ in range(5):
        gpt.generate(encoded_query)
        prof.step()

# the profiler table
prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)

# the profiler trace
prof.export_chrome_trace(trace_path)
