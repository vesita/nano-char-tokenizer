"""Shared Character-level Tokenizer for nanoSeek and DTSeek."""
import os
from typing import List, Optional, Union
from tokenizers import Tokenizer

DEFAULT_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.json")


class NanoCharTokenizer:
    """8192-sparse Character-level Tokenizer based on tokenizers library."""

    def __init__(self, tokenizer_path: Optional[str] = None):
        path = tokenizer_path or DEFAULT_VOCAB_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer json file not found at: {path}")
        self._tokenizer = Tokenizer.from_file(path)
        self.pad_token_id = self._tokenizer.token_to_id("<pad>")
        self.unk_token_id = self._tokenizer.token_to_id("<unk>")
        self.bos_token_id = self._tokenizer.token_to_id("<bos>")
        self.eos_token_id = self._tokenizer.token_to_id("<eos>")
        self.sep_token_id = self._tokenizer.token_to_id("<sep>")
        self.vocab_size = self._tokenizer.get_vocab_size()

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def encode(self, text: str, max_length: int = 1024, padding: bool = True) -> dict:
        """Encode single string into token IDs and attention mask."""
        tokens = self._tokenizer.encode(text)
        ids = tokens.ids[:max_length]
        mask = [1] * len(ids)
        if padding and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            mask = mask + [0] * pad_len
        return {"input_ids": ids, "attention_mask": mask}

    def batch_encode(self, texts: List[str], max_length: int = 1024, padding: bool = True) -> dict:
        batch_ids = []
        batch_mask = []
        for t in texts:
            out = self.encode(t, max_length=max_length, padding=padding)
            batch_ids.append(out["input_ids"])
            batch_mask.append(out["attention_mask"])
        return {"input_ids": batch_ids, "attention_mask": batch_mask}

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self._tokenizer.id_to_token(token_id)


__all__ = ["NanoCharTokenizer"]
