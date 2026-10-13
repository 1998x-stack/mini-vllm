from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    Request,
    TinyModelConfig,
    create_engine,
)


engine = create_engine(
    "v0.13",
    engine_config=EngineConfig(
        max_num_seqs=1,
        max_model_len=48,
        max_num_batched_tokens=4,
        max_speculative_tokens=1,
    ),
    model_config=TinyModelConfig(
        vocab_size=24,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=32,
        max_position_embeddings=64,
        seed=31,
    ),
    cache_config=CacheConfig(block_size=2, num_blocks=32),
    world_size=1,
    max_speculative_tokens=1,
    compile_config=CompileConfig(backend="eager"),
)
core = engine.core
scheduler = core.scheduler

prompt = [1, 2, 3, 4, 5]
core.add_request(Request("seed", prompt, max_new_tokens=1))
core.run()

core.add_request(Request("cold", [9, 10, 11, 12, 13], max_new_tokens=1))
core.add_request(Request("warm", prompt, max_new_tokens=1))
step = core.step()

print("admitted:", step.admitted_request_ids)
print("scheduled:", step.scheduled_request_ids)
print("\nkind       request raw reuse expected bypass protected actual")
for event in scheduler.events:
    if event.request_id not in {"cold", "warm"}:
        continue
    print(
        f"{event.kind:10} {event.request_id:7} "
        f"{event.raw_prefill_tokens:3} {event.reusable_tokens:5} "
        f"{event.expected_new_prefill_tokens:8} {event.bypass_count:6} "
        f"{str(event.fairness_protected):9} {event.actual_reuse_tokens}"
    )

print("\nstats:", scheduler.stats)
