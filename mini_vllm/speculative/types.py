from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DraftBatch:
    token_ids: tuple[int, ...]
    def __post_init__(self) -> None:
        if any(token_id < 0 for token_id in self.token_ids): raise ValueError("draft token ids must be >= 0")
    @property
    def num_tokens(self) -> int: return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    request_id: str
    checkpoint_pos: int
    draft_token_ids: tuple[int, ...]
    accepted_token_ids: tuple[int, ...]
    rejected_token_ids: tuple[int, ...]
    fallback_token_id: int | None
    bonus_token_id: int | None
    rollback_to_pos: int
    @property
    def committed_token_ids(self) -> tuple[int, ...]:
        suffix=()
        if self.fallback_token_id is not None: suffix=(self.fallback_token_id,)
        elif self.bonus_token_id is not None: suffix=(self.bonus_token_id,)
        return self.accepted_token_ids+suffix
    @property
    def fully_accepted(self)->bool: return not self.rejected_token_ids
    @property
    def accepted_count(self)->int: return len(self.accepted_token_ids)

SpeculativeTrace=VerificationResult
