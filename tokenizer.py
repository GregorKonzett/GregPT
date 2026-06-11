import tiktoken


class Tokenizer:
    eos_token_str = "<|g_eos|>"

    def __init__(self):
        base = tiktoken.get_encoding("gpt2")

        self.enc = tiktoken.Encoding(
            name="gpt2_with_eos",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={
                **base._special_tokens,
                self.eos_token_str: base.max_token_value + 1,
            },
        )

        assert self.enc.decode(self.enc.encode("hello world")) == "hello world"

    def encode(self, text):
        return self.enc.encode(text, allowed_special={self.eos_token_str})

    def is_eos(self, token: int) -> bool:
        return token == self.enc.encode(self.eos_token_str, allowed_special={self.eos_token_str})[0]

    def decode(self, arr):
        return self.enc.decode(arr)

    def get_vocab_size(self):
        return self.enc.max_token_value + 1



