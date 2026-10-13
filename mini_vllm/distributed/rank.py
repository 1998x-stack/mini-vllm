from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankConfig:
    """Explicit logical-rank ownership for educational distributed execution."""

    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
