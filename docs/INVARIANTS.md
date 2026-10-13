# mini-vLLM v1.2 invariant contract

v1.2 preserves all v1.1 serving/process semantics and adds the trust boundary from external model/tokenizer files to validated runtime-owned state.

## Logical token state

1. `num_computed_tokens` counts only tokens whose model work executed and whose K/V remains valid and addressable.
2. A sampled output token is logical history, not computed history, until a later forward executes it.
3. Preemption may preserve logical history while rolling the computed frontier backward.
4. Text is converted to token IDs before entering the serving core; Scheduler/KV/WorkItem contracts remain token-native.

## KV ownership and addressing

5. Physical ownership/reservation exists before any K/V read or write.
6. Paged translation is authoritative:

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[logical_block]
```

7. Attention reads only logical positions `[0, length)`.
8. Shared prefix blocks are immutable while referenced.
9. Finished/cancelled requests release active ownership exactly once.
10. Physical KV/cache structures remain child-owned; the parent never treats block tables/refcounts/cache objects as remote ownership handles.

## Prefix reuse, preemption and speculation

11. `PrefixReuseHint` is advisory, never reservation/ownership/computed state.
12. Side-effect-free prefix estimation cannot create ownership or mutate cache recency.
13. Only authoritative prepare establishes the actual reusable computed frontier.
14. Warm-prefix preference retains deterministic starvation protection.
15. A preempted request cannot claim released KV.
16. Replay preserves known logical history without resampling historical outputs.
17. Rejected speculative state disappears from logical history and physical KV.
18. Speculative rollback never mutates shared immutable prefixes.

## External artifact truth

19. Filesystem presence is not proof of a valid model artifact.
20. Successful JSON parsing is not proof of schema/architecture compatibility.
21. Successful checkpoint deserialization is not proof of complete or compatible model state.
22. Artifact format/version and architecture must be accepted explicitly before runtime construction.
23. Required manifest fields are exact; missing or unexpected incompatible fields fail explicitly.
24. Model dimensions must satisfy the same `TinyModelConfig` validity rules as programmatic construction.
25. The artifact path is location, not model identity.

## Weight manifest and copy

26. The external tensor-name set must exactly equal the required mapping set for the reference format.
27. Missing required tensors are never silently initialized.
28. Unexpected tensors are never silently ignored.
29. Every accepted source tensor must match its expected shape before copy.
30. Every accepted source tensor must use a supported artifact dtype before copy.
31. Non-tensor checkpoint entries are rejected.
32. The external-name → runtime-name relationship remains explicit even when names are identical.
33. Every required runtime tensor target is populated exactly once under the accepted mapping.
34. Runtime model state owns copied tensor values; serving correctness cannot depend on retaining or mutating the loader checkpoint dictionary.
35. Runtime dtype/device conversion may change representation after validation but cannot change model semantics outside documented numeric behavior.

## Tokenizer compatibility

36. Tokenizer format/type/version are validated before use.
37. Tokenizer tokens are non-empty and unique under the reference format.
38. Required special-token metadata, including `unk`, must contain in-range integer IDs.
39. `tokenizer.vocab_size == model.vocab_size` before readiness.
40. tokenizer compatibility ID in `tokenizer.json` equals the model manifest compatibility ID.
41. Unknown-text handling is deterministic according to the declared `unk` token.
42. Decode rejects token IDs outside the tokenizer vocabulary.

## Artifact identity and multiprocess readiness

43. `ModelArtifactIdentity` is derived from format/architecture plus model, weights and tokenizer digests; it is not a path or mutable object reference.
44. Parent validation establishes the exact identity expected by one engine startup.
45. Every child independently re-reads/re-validates its configured artifact before reporting ready.
46. A child that observes changed bytes after parent validation must fail identity comparison rather than silently adopting the new artifact.
47. Every ready rank reports the parent-validated artifact identity.
48. Every ready rank reports the same tokenizer identity.
49. Executor readiness requires cross-rank identity agreement before any request admission/execution.
50. If one worker fails artifact startup, already-created workers are shut down/reaped; partial startup cannot leak orphan processes.
51. A default temporary educational artifact follows the same loader/child-revalidation path as an explicit artifact; it is not a serving bypass.
52. Runtime-owned temporary artifact files are removed only after workers stop.

## Multiprocessing execution

53. EngineCore owns logical commit decisions; child Worker/ModelRunner owns physical execution.
54. IPC contains serializable logical values/copies, not live child model/cache/tensor-storage references.
55. Every serving command/result uses a matching correlation ID.
56. Sending a command is not evidence of success; logical commit requires a matching successful acknowledgment.
57. Child-reported execution errors and transport/process-death failures remain distinct.
58. Parent-side checkpoint handles identify child-owned transaction state; they are not KV ownership handles.
59. Failed ordinary execution attempts rollback on every reachable checkpointed participant before surfacing failure.
60. Unknown physical consistency after EOF/timeout/process death/rollback failure makes the executor unhealthy; EngineCore cannot guess and commit.
61. Successful close/aclose joins/reaps every child process.

## Distribution and optimization

62. Replicated ranks agree on policy-visible cache facts and model identity required by the operation.
63. Tensor-parallel gathered logits remain equivalent to the supported readable reference.
64. Vectorized PagedAttention is numerically equivalent to reference PagedAttention.
65. Compile-aware LM-head projection is numerically equivalent to eager projection.
66. Optimization fallback may change execution mode, never serving semantics.
67. No speedup claim follows merely from multiprocessing, loading, vectorization or compilation.

## Async delivery and observability

68. Async clients observe committed tokens only.
69. Each request exposes exactly one terminal outcome.
70. Backpressure may delay delivery but cannot duplicate or hide logical commits.
71. Cancellation is applied between atomic EngineCore steps and does not arbitrarily kill healthy workers.
72. `snapshot()` is read-only: it cannot schedule, attach KV, mutate recency, advance requests or alter serving-command counters.
73. Artifact observations expose logical identity/configuration facts, never checkpoint dictionaries, parameter pointers or storage ownership handles.
74. Repeated snapshots with no intervening serving action remain equal.

## Semantic equivalence

75. Exporting the deterministic v1.1 TinyDecoder into the v1.2 reference artifact and loading it back must reproduce the same committed output under equivalent world-size 1 and 2 serving configuration.
76. Historical v0.1–v1.1 builders retain their original mechanism semantics; v1.2 artifact loading is opt-in only through the v1.2/default builder.
