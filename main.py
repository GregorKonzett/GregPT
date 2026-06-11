from DatasetLoader import DatasetLoader
from GptModel import GptModel, get_device
from GptTrainer import GptTrainer
import torch
from tokenizer import Tokenizer
import getopt, sys

tokenizer = Tokenizer()
gpt = GptModel(vocab_size=tokenizer.get_vocab_size(), tokenizer=tokenizer)
trainer = GptTrainer(gpt)

args = sys.argv[1:]
long_options = ["pre=", "post=", "infer="]

try:
    arguments, values = getopt.getopt(args, "", long_options)

    for currentArgument, currentValue in arguments:
        if currentArgument == "--pre":
            dataset_loader = DatasetLoader("Salesforce/wikitext", "wikitext-103-raw-v1")
            iters = int(currentValue)
            data = dataset_loader.get_train_data()
            training_data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
            training_length = int(len(training_data) * 0.9)
            print(f"Pre-training with input: {len(data)} {data[:100]}")
            trainer.pre_train(iters, training_data[:training_length], training_data[training_length:], True)
        elif currentArgument == "--post":
            iters = int(currentValue)
            print(f"Post training: {iters}")
        elif currentArgument == "--infer":
            query = currentValue
            print(f"Inference with input: {query}")
            trainer.restore_checkpoint()
            encoded_query = torch.tensor([tokenizer.encode(query)], dtype=torch.long, device=get_device())
            out = gpt.generate(encoded_query)
            print("Output:")
            print(tokenizer.decode(out[0].cpu().tolist()))
except getopt.error as err:
    print(str(err))
