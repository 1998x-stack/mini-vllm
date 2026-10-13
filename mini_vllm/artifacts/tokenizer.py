from __future__ import annotations

from dataclasses import dataclass

from mini_vllm.artifacts.types import TokenizerArtifact


@dataclass(frozen=True, slots=True)
class WhitespaceTokenizer:
    """Small deterministic tokenizer used to teach the text/token boundary.

    Encoding splits on Unicode whitespace and maps whole pieces through the
    artifact vocabulary. Unknown pieces map to the declared ``unk`` token.
    Decoding joins vocabulary entries with one ASCII space.
    """

    artifact: TokenizerArtifact

    def __post_init__(self) -> None:
        if self.artifact.tokenizer_type != "whitespace":
            raise ValueError(
                f"unsupported tokenizer type: {self.artifact.tokenizer_type}"
            )

    @property
    def vocab_size(self) -> int:
        return self.artifact.vocab_size

    @property
    def identity(self) -> str:
        return self.artifact.digest

    @property
    def compatibility_id(self) -> str:
        return self.artifact.compatibility_id

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        pieces = text.split()
        if not pieces:
            raise ValueError("text must contain at least one token")
        lookup = {token: index for index, token in enumerate(self.artifact.tokens)}
        special = self.artifact.special_token_map()
        unk = special.get("unk")
        encoded: list[int] = []
        for piece in pieces:
            token_id = lookup.get(piece)
            if token_id is None:
                if unk is None:
                    raise ValueError(f"tokenizer has no mapping for {piece!r}")
                token_id = unk
            encoded.append(token_id)
        return encoded

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        if not token_ids:
            return ""
        tokens: list[str] = []
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("token ids must be integers")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(
                    f"token id {token_id} outside tokenizer vocabulary 0..{self.vocab_size - 1}"
                )
            tokens.append(self.artifact.tokens[token_id])
        return " ".join(tokens)
