from mini_vllm.model.runner import ModelRunner, ScriptedModelRunner, TinyTransformerRunner
from mini_vllm.model.tiny_transformer import TinyDecoderLM

__all__ = [
    "ModelRunner",
    "ScriptedModelRunner",
    "TinyTransformerRunner",
    "VocabParallelTinyRunner",
    "TinyDecoderLM",
]


def __getattr__(name: str):
    if name == "VocabParallelTinyRunner":
        from mini_vllm.model.tensor_parallel_runner import VocabParallelTinyRunner

        return VocabParallelTinyRunner
    raise AttributeError(name)
