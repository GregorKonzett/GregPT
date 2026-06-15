import tiktoken


class TikTokenTokenizer:
    eos_token_str = "<|g_eos|>"
    system_token_str = "<|system|>"
    user_token_str = "<|user|>"
    assistant_token_str = "<|assistant|>"
    end_of_text_str = "<|endoftext|>"

    special_token_strs = {
        eos_token_str,
        system_token_str,
        user_token_str,
        assistant_token_str,
    }

    def __init__(self):
        base = tiktoken.get_encoding("gpt2")

        self.enc = tiktoken.Encoding(
            name="gpt2_custom",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={
                **base._special_tokens,
                self.eos_token_str: base.max_token_value + 1,
                self.system_token_str: base.max_token_value + 2,
                self.user_token_str: base.max_token_value + 3,
                self.assistant_token_str: base.max_token_value + 4,
            },
        )

        self.eos_token = self.enc.encode(self.eos_token_str, allowed_special=self.special_token_strs)[0]

        assert self.enc.decode(self.enc.encode("hello world")) == "hello world"

    def encode(self, text):
        return self.enc.encode(
            text,
            allowed_special=self.special_token_strs,
            disallowed_special=self.enc.special_tokens_set - self.special_token_strs - {self.end_of_text_str}
        )

    def get_eos_token(self):
        return self.eos_token

    def is_eos(self, token: int) -> bool:
        return token == self.eos_token

    def decode(self, arr):
        return self.enc.decode(arr)

    def get_vocab_size(self):
        return self.enc.max_token_value + 1



