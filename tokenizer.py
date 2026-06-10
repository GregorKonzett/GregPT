import tiktoken


class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        assert self.enc.decode(self.enc.encode("hello world")) == "hello world"

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, arr):
        return self.enc.decode(arr)

    def get_vocab_size(self):
        return self.enc.max_token_value

