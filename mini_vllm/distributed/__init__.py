from mini_vllm.distributed.collective import ReferenceCollective
from mini_vllm.distributed.rank import RankConfig
from mini_vllm.distributed.tensor_parallel import (
    ReferenceTensorParallel,
    VocabParallelLMHead,
    VocabShard,
    shard_range,
)

__all__ = [
    "RankConfig",
    "ReferenceCollective",
    "ReferenceTensorParallel",
    "VocabParallelLMHead",
    "VocabShard",
    "shard_range",
    "RankedWorker",
    "WorkerGroup",
    "DistributedCacheCheckpoint",
    "DistributedExecutor",
]


def __getattr__(name: str):
    """Lazy-load execution-layer distributed objects to keep leaf imports acyclic."""
    if name == "DistributedExecutor":
        from mini_vllm.distributed.executor import DistributedExecutor

        return DistributedExecutor
    if name in {"RankedWorker", "WorkerGroup", "DistributedCacheCheckpoint"}:
        from mini_vllm.distributed.worker_group import (
            DistributedCacheCheckpoint,
            RankedWorker,
            WorkerGroup,
        )

        return {
            "RankedWorker": RankedWorker,
            "WorkerGroup": WorkerGroup,
            "DistributedCacheCheckpoint": DistributedCacheCheckpoint,
        }[name]
    raise AttributeError(name)
