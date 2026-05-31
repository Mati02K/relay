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
    slow_queue = _worker("busy", qw=20)
    low_queue = _worker("idle", qw=1)

    choice = choose_worker(request, [slow_queue, low_queue])

    assert choice.worker.node_id == "idle"


def test_choose_worker_prefers_longer_prefix_overlap_when_load_is_equal() -> None:
    content = " ".join(["shared-prefix"] * 80)
    request = _request(content)
    prompt_text = request_to_prefix_text(request)
    prefix = compute_prefix_hashes_from_text(prompt_text)

    cold = _worker("cold", qw=0)
    warm = _worker(
        "warm",
        qw=0,
        prefix_cache=PrefixCacheTelemetry.from_config(prefix.config, prefix.block_hashes),
    )

    choice = choose_worker(request, [cold, warm])

    assert choice.worker.node_id == "warm"
    assert choice.overlap > 0.9
    assert choice.uncached_prompt_tokens == choice.prompt_tokens - choice.matched_tokens


def test_choose_worker_positive_weight_flips_tied_workers() -> None:
    request = _request("hello")
    plain = _worker("plain", qw=5)
    preferred = _worker("preferred", qw=5, weight=0.5)

    choice = choose_worker(request, [plain, preferred])

    assert choice.worker.node_id == "preferred"


def test_choose_worker_negative_weight_deprioritizes() -> None:
    request = _request("hello")
    plain = _worker("plain", qw=5)
    penalised = _worker("penalised", qw=5, weight=-0.5)

    choice = choose_worker(request, [plain, penalised])

    assert choice.worker.node_id == "plain"


def test_choose_worker_weight_clamped_at_boundaries() -> None:
    request = _request("hello")
    # Both idle, so only worker_weight differs. An out-of-range weight 5.0 is
    # clamped to the +1.0 cap, so it only *ties* the worker already at the cap
    # rather than beating it — proven by the lower node id winning the tie. An
    # unclamped 5.0 would have made "zzz" the cheapest and won outright.
    maxed = _worker("aaa", qw=0, weight=1.0)
    over_cap = _worker("zzz", qw=0, weight=5.0)

    choice = choose_worker(request, [maxed, over_cap])

    assert choice.worker.node_id == "aaa"


def _request(content: str) -> dict[str, object]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": content}],
    }


def _worker(
    node_id: str,
    *,
    qw: int,
    prefix_cache: PrefixCacheTelemetry | None = None,
    weight: float = 0.0,
) -> WorkerSnapshot:
    telemetry = Telemetry(
        qw=qw,
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
            ],
            "weight": weight,
        },
        telemetry=telemetry,
    )
