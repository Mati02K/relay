from coordinator.scheduler import choose_worker
from coordinator.worker_registry import WorkerSnapshot
from telemetry.prefix_cache import (
    PrefixHashConfig,
    compute_prefix_hashes_from_text,
    request_to_prefix_text,
)
from telemetry.schemas import PrefixCacheTelemetry, Telemetry


def test_choose_worker_prefers_lower_queue_over_equal_workers() -> None:
    request = _request("Explain Relay scheduling in one sentence.")
    slow_queue = _worker("busy", qw=20, decode_speed=10.0)
    low_queue = _worker("idle", qw=1, decode_speed=10.0)

    choice = choose_worker(request, [slow_queue, low_queue])

    assert choice.worker.node_id == "idle"


def test_choose_worker_prefers_longer_prefix_overlap_when_load_is_equal() -> None:
    content = " ".join(["shared-prefix"] * 80)
    request = _request(content)
    prompt_text = request_to_prefix_text(request)
    prefix = compute_prefix_hashes_from_text(prompt_text)

    cold = _worker("cold", qw=0, decode_speed=10.0)
    warm = _worker(
        "warm",
        qw=0,
        decode_speed=10.0,
        prefix_cache=PrefixCacheTelemetry.from_config(prefix.config, prefix.block_hashes),
    )

    choice = choose_worker(request, [cold, warm])

    assert choice.worker.node_id == "warm"
    assert choice.overlap > 0.9
    assert choice.uncached_prompt_tokens == choice.prompt_tokens - choice.matched_tokens


def _request(content: str) -> dict[str, object]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": content}],
    }


def _worker(
    node_id: str,
    *,
    qw: int,
    decode_speed: float,
    prefix_cache: PrefixCacheTelemetry | None = None,
) -> WorkerSnapshot:
    telemetry = Telemetry(
        qw=qw,
        sw_by_bucket={"<=256": decode_speed, "<=1024": decode_speed},
        mw=0.0,
        jw=0.0,
        theta_w=0,
        prefix_cache=prefix_cache
        or PrefixCacheTelemetry.from_config(PrefixHashConfig(model_id="test-model"), []),
        sprefill_tokens_per_sec=0.0,
    )
    return WorkerSnapshot(
        node_id=node_id,
        address=f"http://{node_id}",
        metadata={
            "models": [
                {
                    "id": "test-model",
                    "loaded": True,
                }
            ]
        },
        telemetry=telemetry,
    )
