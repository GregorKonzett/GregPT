import re

from datasets import load_dataset

from tokenizer import Tokenizer


class DatasetLoader:
    def __init__(self, path, name = None):
        self.path = path
        self.name = name

    def __remove_equals(self, line: str) -> str:
        newline = ""
        if line.endswith("\r\n"):
            line = line[:-2]
            newline = "\r\n"
        elif line.endswith("\n"):
            line = line[:-1]
            newline = "\n"

        line = line.strip()
        line = re.sub(r"^(?:= )+", "", line)
        line = re.sub(r"(?: =)+$", "", line)

        return line.strip() + newline

    def __clean_artifacts(self, line: str) -> str:
        line = re.sub(r"\s*@-@\s*", "-", line)
        line = re.sub(r"\s*@,@\s*", ",", line)
        line = re.sub(r"\s*@\.@\s*", ".", line)
        line = line.strip()
        line = re.sub(r"\b([A-Za-z0-9]+)\s+'(s|re|ve|ll|d|m|t)\b", r"\1'\2", line)
        line = re.sub(r"\b([A-Za-z]+)\s+n't\b", r"\1n't", line)
        line = re.sub(r"\s+([,.;:!?%])", r"\1", line)
        line = re.sub(r"\s+([\)\]\}])", r"\1", line)
        line = re.sub(r"([\(\[\{])\s+", r"\1", line)
        line = line.replace("`` ", '"').replace(" ''", '"')

        return line

    def __prepare_pre_data(self, text) -> str:
        out = []
        skip = True

        for line in text:
            stripped_line = line.strip()
            is_title = (stripped_line.startswith("= ")
                        and stripped_line.endswith(" =")
                        and not stripped_line.endswith(" = ="))

            is_subtitle = (stripped_line.startswith("= = ") and stripped_line.endswith(" = ="))

            if is_title and not skip:
                out.append(Tokenizer.eos_token_str)
            elif is_title:
                skip = False

            if is_title or is_subtitle:
                line = self.__remove_equals(line)

            cleaned = self.__clean_artifacts(line)

            if cleaned:
                out.append(cleaned)

        # Add one after the last article
        out.append(Tokenizer.eos_token_str)

        return '\n'.join(out)

    def __prepare_post_data(self, content) -> str:
        out = []

        for chat in content:
            full_out = ""

            for turn in chat:
                if turn["role"] == "system":
                    full_out += Tokenizer.system_token_str  + '\n'
                elif turn["role"] == "user":
                    full_out += Tokenizer.user_token_str  + '\n'
                else:
                    full_out += Tokenizer.assistant_token_str  + '\n'

                full_out += turn["content"] + '\n'

            full_out += Tokenizer.eos_token_str

            out.append(full_out)

        return '\n'.join(out)

    def __prepare_data(self, phase, data) -> str:
        if phase == "pre":
            return self.__prepare_pre_data(data[["text"]])
        elif phase == "post":
            return self.__prepare_post_data(data["messages"])
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def get_train_data(self, phase) -> str:
        pass

    def get_val_data(self, phase, split) -> str:
        pass

    def download_data(self, phase, split) -> str:
        print(f"Downloading {split} {self.name} dataset from {self.path}")
        ds = load_dataset(self.path, self.name)
        text = self.__prepare_data(phase, ds[split])

        return text
