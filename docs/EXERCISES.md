# mini-vLLM v1.2 exercises

Use historical diffs to learn one mechanism at a time. v1.2 focuses on the boundary between external model files and trusted runtime-owned serving state.

## Artifact truth

1. Run `examples/17_model_loading.py`. Draw the path `model directory → validated artifact → child runtime model → request output`.
2. Explain why a model directory existing on disk is weaker evidence than `ValidatedModelArtifact`.
3. Change the artifact format version. Identify the exact layer that rejects it and why no worker/request should exist afterward.
4. Add an unknown field to `model_config.json`. Why is strict schema rejection useful in an educational serving runtime?

## Identity

5. List the components of `ModelArtifactIdentity`. Why is the filesystem path absent from the digest contract?
6. Export the same deterministic model twice. Compare identities.
7. Change one model weight and re-load. Which digest must change?
8. Change only tokenizer metadata. Which identity components must change?

## Weight mapping and coverage

9. Inspect `build_weight_mappings()`. For one parameter, record external name, runtime name, expected shape and dtype.
10. Delete one checkpoint tensor. Why is deterministic initialization of the missing parameter forbidden in this format?
11. Add one unexpected tensor. Why is ignoring it silently a compatibility hazard?
12. Preserve a tensor name but change its shape. Explain why name equality is insufficient.
13. Convert one tensor to float64. Where is artifact dtype policy enforced?
14. Put a non-tensor object into the checkpoint. Why must it never become runtime parameter state?

## Runtime ownership

15. Follow one accepted source tensor into `build_model_from_artifact()` and identify the explicit copy point.
16. After model construction, mutate the loader-side tensor container. Prove the runtime parameter does not change.
17. Explain the difference between artifact weight dtype and runtime dtype/device.
18. Why does a runtime-owned copy make loader/container lifetime irrelevant after startup?

## Tokenizer boundary

19. Encode `tok1 tok2 unknown`. Explain the deterministic unknown-token result.
20. Try to decode an out-of-range token ID. Why is rejection better than string fallback?
21. Make tokenizer vocabulary size differ from model vocabulary size. Why must startup fail before text serving?
22. Trace `submit_text()`. At what exact point do strings disappear from the system?
23. List the core layers that must remain unaware of text: Scheduler, WorkItem, KV, ModelRunner and speculation.

## Parent versus child validation

24. Parent-load an artifact, then change a checkpoint tensor before worker startup. Why must child re-validation fail identity comparison?
25. Explain why sending the parent checkpoint dictionary through IPC would teach a second large mechanism that v1.2 deliberately avoids.
26. For world size 2, list the evidence required before `ArtifactMultiprocessExecutor` can become ready.
27. Construct two `WorkerSpec`s with different artifact identities. Why should the parent reject them before process startup?
28. Make rank 1 fail loading after rank 0 is ready. What must happen to rank 0 before construction returns an error?

## Readiness and serving

29. Expand the v1.2 readiness equation: process alive + artifact valid + weights copied + tokenizer compatible + identity agreement.
30. Why must no request be admitted while any rank is still loading or validating its model?
31. After readiness, which v1.1 serving algorithms are unchanged?
32. Explain why artifact identity should not become a Scheduler ranking input.

## Default builder

33. Call `create_engine()` with no artifact path. Find the temporary model directory.
34. Explain why exporting a temporary artifact and loading it back is stronger than keeping a direct in-memory TinyDecoder bypass.
35. Close the engine and verify the temporary directory disappears only after workers stop.

## Observability

36. Inspect `snapshot.core.model_artifact`. Explain every field and which ones are logical facts rather than storage handles.
37. Compare every rank's `artifact_identity` and `tokenizer_identity`.
38. Call `snapshot()` twice without serving work. Prove artifact observations preserve the read-only snapshot invariant.
39. Why are raw checkpoint dictionaries and parameter pointers intentionally absent?

## Semantic equivalence

40. Export the deterministic v1.1 model, run v1.1 and v1.2 at world size 1, and compare committed outputs.
41. Repeat at world size 2. What does equality prove, and what does it not prove?
42. Explain why identical token output does not by itself prove strict artifact validation; which negative tests are still needed?

## Failure taxonomy

43. Distinguish parent artifact validation failure from child startup identity mismatch.
44. Distinguish child startup load failure from a post-readiness `WorkerRemoteError` during execution.
45. Why does a malformed artifact not need execution rollback, while a failed serving step may?
46. Explain how v1.2 artifact trust and v1.1 transport acknowledgment form two independent correctness boundaries.

## Historical synthesis

47. Compare v1.0, v1.1 and v1.2: composition → process isolation → external-artifact ingestion.
48. Compare v0.8's class-level Worker boundary with v1.1's process boundary and v1.2's artifact boundary.
49. Explain why adding safetensors support alone would usually be an incremental v1.2 extension rather than a new learning version.
50. Propose a v1.3 network/API boundary. Identify what new owner appears at the edge and which existing invariants must survive it.
