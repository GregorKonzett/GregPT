from dataset.GCPStorageDatasetLoader import GCPStorageDatasetLoader
from GptModel import GptModel, get_device
from GptTrainer import GptTrainer
import torch
from tokenizer import Tokenizer
import getopt, sys

from weights.GCPStorageWeightLoader import GCPStorageWeightLoader

tokenizer = Tokenizer()
weight_loader = GCPStorageWeightLoader()
gpt = GptModel(vocab_size=tokenizer.get_vocab_size(), tokenizer=tokenizer)
trainer = GptTrainer(gpt, weight_loader, tokenizer)

args = sys.argv[1:]
long_options = ["pre=", "post=", "infer="]

try:
    arguments, values = getopt.getopt(args, "", long_options)

    for currentArgument, currentValue in arguments:
        if currentArgument == "--pre":
            dataset_loader = GCPStorageDatasetLoader("Salesforce/wikitext", "wikitext-103-raw-v1")
            iters = int(currentValue)
            training_data = torch.tensor(tokenizer.encode(dataset_loader.get_train_data("pre")), dtype=torch.long)
            eval_data = torch.tensor(tokenizer.encode(dataset_loader.get_val_data("pre", "validation")), dtype=torch.long)
            print(f"Pre-training with input: {len(training_data)} {training_data[:100]}")
            trainer.pre_train(iters, training_data, eval_data, True)
        elif currentArgument == "--post":
            dataset_loader = GCPStorageDatasetLoader("HuggingFaceTB/smol-smoltalk")
            iters = int(currentValue)
            train_data_list = [
                data for data in dataset_loader.get_train_data("post").split(Tokenizer.eos_token_str)
                if data.strip()
            ]
            training_data = [
                torch.tensor(tokenizer.encode(data + Tokenizer.eos_token_str), dtype=torch.long) for data in train_data_list
            ]
            test_data_list = [
                data for data in dataset_loader.get_val_data("post", "test").split(Tokenizer.eos_token_str)
                if data.strip()
            ]
            test_data = [
                torch.tensor(tokenizer.encode(data + Tokenizer.eos_token_str), dtype=torch.long) for data in test_data_list
            ]

            trainer.post_train(iters, training_data, test_data, True)
        elif currentArgument == "--infer":
            query = currentValue
            print(f"Inference with input: {query}")
            weight_loader.load_checkpoint(gpt)
            encoded_query = torch.tensor([tokenizer.encode(query)], dtype=torch.long, device=get_device())
            out = gpt.generate(encoded_query)
            print("Output:")
            print(tokenizer.decode(out[0].cpu().tolist()))
except getopt.error as err:
    print(str(err))
