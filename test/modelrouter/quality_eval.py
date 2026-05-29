"""Side-by-side answer quality comparison between Gemma (strong) and TinyLlama (weak).

For each prompt we:

1. Fire the same prompt directly at worker-a (Gemma-3-4B-it) and worker-b
   (TinyLlama 1.1B), bypassing the coordinator so the comparison is purely
   a model-vs-model evaluation, not a routing evaluation.
2. Compute simple objective text metrics on both answers.
3. Ask Gemma to judge which of the two answers better addresses the prompt,
   with the order of (A, B) randomized per prompt so the judge cannot tell
   which model produced which response just from position.

LLM-as-judge has known biases (the judge tends to favor its own writing
style, verbosity, and so on); we run Gemma as the judge here only because
no stronger model is available locally. The objective metrics are reported
alongside so the verdict can be sanity-checked.

Output:

* ``bench/results/quality_eval.json`` — raw per-prompt record
* ``bench/results/quality_eval.md`` — human-readable summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx


WORKER_STRONG_URL = "http://127.0.0.1:9090"   # Gemma-3-4B-it
WORKER_WEAK_URL = "http://127.0.0.1:9091"     # TinyLlama 1.1B

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge comparing two answers to the same user question. "
    "Pick the answer that is more accurate, more coherent, and more directly "
    "addresses the question. Briefly justify in one sentence, then on a new "
    "line output exactly 'WINNER: A' or 'WINNER: B' or 'WINNER: TIE'."
)


@dataclass
class Answer:
    """One model's response to one prompt."""

    worker_url: str
    text: str
    completion_tokens: int
    prompt_tokens: int
    decode_seconds: float


@dataclass
class JudgeVerdict:
    """Outcome of the LLM-as-judge call."""

    raw: str
    winner: str  # "strong", "weak", "tie", or "unparsed"
    swapped_for_judging: bool


@dataclass
class PromptResult:
    """Everything observed for one evaluation prompt."""

    prompt_id: str
    prompt_text: str
    strong: Answer
    weak: Answer
    metrics: dict
    judge: JudgeVerdict | None


async def _generate(client: httpx.AsyncClient, worker_url: str, prompt: str,
                    max_tokens: int = 200) -> Answer:
    """Call a worker's ``/v1/generate`` and collect the final assistant text."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    pieces: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    decode_s = 0.0
    async with client.stream(
        "POST", f"{worker_url}/v1/generate",
        json=body, headers={"Accept": "text/event-stream"}, timeout=120,
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ").strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    pieces.append(delta["content"])
            usage = obj.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                completion_tokens = usage.get("completion_tokens") or completion_tokens
            timings = obj.get("timings")
            if isinstance(timings, dict):
                predicted_ms = timings.get("predicted_ms")
                if isinstance(predicted_ms, (int, float)) and predicted_ms > 0:
                    decode_s = predicted_ms / 1000.0
    return Answer(
        worker_url=worker_url,
        text="".join(pieces).strip(),
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        decode_seconds=decode_s,
    )


def _text_metrics(text: str) -> dict:
    """Cheap, deterministic, judge-free text quality signals."""
    words = re.findall(r"\w+", text.lower())
    n = len(words)
    unique = len(set(words))
    bigrams = list(zip(words[:-1], words[1:]))
    bigram_counts = Counter(bigrams)
    repeated_bigram_fraction = (
        sum(c for c in bigram_counts.values() if c > 1) / max(1, len(bigrams))
    )
    return {
        "char_count": len(text),
        "word_count": n,
        "unique_word_ratio": round(unique / n, 3) if n else 0.0,
        "repeated_bigram_fraction": round(repeated_bigram_fraction, 3),
    }


async def _judge(client: httpx.AsyncClient, prompt: str, ans_a: str, ans_b: str) -> str:
    """Ask Gemma which of two anonymized answers is better."""
    judge_prompt = (
        f"Question:\n{prompt}\n\n"
        f"Answer A:\n{ans_a}\n\n"
        f"Answer B:\n{ans_b}\n\n"
        "Which answer is better?"
    )
    body = {
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt},
        ],
        "max_tokens": 120,
        "temperature": 0.0,
    }
    pieces: list[str] = []
    async with client.stream(
        "POST", f"{WORKER_STRONG_URL}/v1/generate",
        json=body, headers={"Accept": "text/event-stream"}, timeout=120,
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ").strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    pieces.append(delta["content"])
    return "".join(pieces).strip()


def _parse_winner(raw: str, swapped: bool) -> str:
    """Translate the judge's 'WINNER: A/B/TIE' into 'strong'/'weak'/'tie'."""
    upper = raw.upper()
    if "WINNER: TIE" in upper:
        return "tie"
    # If A was the strong response, "WINNER: A" -> strong. If swapped, flip.
    if "WINNER: A" in upper:
        return "weak" if swapped else "strong"
    if "WINNER: B" in upper:
        return "strong" if swapped else "weak"
    return "unparsed"


async def evaluate(prompt_id: str, prompt_text: str, rng: random.Random,
                   max_tokens: int, do_judge: bool) -> PromptResult:
    """Run one prompt against both models, judge if requested."""
    async with httpx.AsyncClient() as client:
        strong, weak = await asyncio.gather(
            _generate(client, WORKER_STRONG_URL, prompt_text, max_tokens),
            _generate(client, WORKER_WEAK_URL, prompt_text, max_tokens),
        )
        metrics = {
            "strong": _text_metrics(strong.text),
            "weak": _text_metrics(weak.text),
        }
        verdict: JudgeVerdict | None = None
        if do_judge:
            swapped = rng.random() < 0.5
            a, b = (weak.text, strong.text) if swapped else (strong.text, weak.text)
            raw = await _judge(client, prompt_text, a, b)
            verdict = JudgeVerdict(
                raw=raw,
                winner=_parse_winner(raw, swapped),
                swapped_for_judging=swapped,
            )
        return PromptResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            strong=strong,
            weak=weak,
            metrics=metrics,
            judge=verdict,
        )


def _pick_prompts(all_prompts: list[dict], rng: random.Random) -> list[dict]:
    """Select two short, two medium, two long prompts deterministically."""
    sorted_by_len = sorted(all_prompts, key=lambda p: len(p["content"]))
    n = len(sorted_by_len)
    short = sorted_by_len[: n // 3]
    medium = sorted_by_len[n // 3 : 2 * n // 3]
    long_ = sorted_by_len[2 * n // 3 :]
    return rng.sample(short, 2) + rng.sample(medium, 2) + rng.sample(long_, 2)


def _truncate(s: str, n: int = 400) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n].rstrip() + "..."


def _write_markdown(results: list[PromptResult], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Quality evaluation: Gemma-3-4B-it vs TinyLlama 1.1B")
    lines.append("")
    lines.append(
        f"Each prompt was sent directly to worker-a (Gemma) and worker-b (TinyLlama), "
        f"bypassing the coordinator. {sum(1 for r in results if r.judge)} of "
        f"{len(results)} pairs were judged by Gemma itself with the order randomized."
    )
    lines.append("")
    judged = [r for r in results if r.judge]
    if judged:
        wins = sum(1 for r in judged if r.judge and r.judge.winner == "strong")
        losses = sum(1 for r in judged if r.judge and r.judge.winner == "weak")
        ties = sum(1 for r in judged if r.judge and r.judge.winner == "tie")
        unparsed = sum(1 for r in judged if r.judge and r.judge.winner == "unparsed")
        lines.append(
            f"## Headline verdict\n\n"
            f"- Strong (Gemma) wins: **{wins}/{len(judged)}**\n"
            f"- Weak (TinyLlama) wins: {losses}/{len(judged)}\n"
            f"- Ties: {ties}/{len(judged)}\n"
            f"- Unparsed: {unparsed}/{len(judged)}\n\n"
            f"LLM-as-judge bias caveat: the judge is the same model as the strong "
            f"competitor, so a self-preference effect is possible. Use the objective "
            f"metrics below to sanity-check.\n"
        )

    lines.append("## Objective metrics (no judge needed)")
    lines.append("")
    lines.append("| prompt | chars | strong words | strong unique-word ratio | strong repeated-bigram frac | weak words | weak unique-word ratio | weak repeated-bigram frac |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        ms, mw = r.metrics["strong"], r.metrics["weak"]
        lines.append(
            f"| {r.prompt_id} | {len(r.prompt_text)} | "
            f"{ms['word_count']} | {ms['unique_word_ratio']:.2f} | {ms['repeated_bigram_fraction']:.2f} | "
            f"{mw['word_count']} | {mw['unique_word_ratio']:.2f} | {mw['repeated_bigram_fraction']:.2f} |"
        )
    lines.append("")

    lines.append("## Side-by-side answers")
    for r in results:
        lines.append(f"\n### `{r.prompt_id}` (prompt length {len(r.prompt_text)} chars)")
        lines.append("")
        lines.append(f"**Prompt:** {_truncate(r.prompt_text, 240)}")
        lines.append("")
        lines.append(f"**Strong (Gemma):** {_truncate(r.strong.text, 600)}")
        lines.append("")
        lines.append(f"**Weak (TinyLlama):** {_truncate(r.weak.text, 600)}")
        if r.judge:
            lines.append("")
            verdict_label = {
                "strong": "Gemma wins",
                "weak": "TinyLlama wins",
                "tie": "tie",
                "unparsed": "unparsed",
            }[r.judge.winner]
            lines.append(
                f"**Judge verdict:** {verdict_label} (swapped={r.judge.swapped_for_judging})"
            )
            lines.append(f"> {_truncate(r.judge.raw, 400)}")

    out_path.write_text("\n".join(lines) + "\n")


def _serialize(results: list[PromptResult]) -> list[dict]:
    out: list[dict] = []
    for r in results:
        d = asdict(r)
        # Trim the raw judge text only for the json copy if it's huge
        out.append(d)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("bench/data/prompts.jsonl"))
    parser.add_argument("--out-md", type=Path, default=Path("bench/results/quality_eval.md"))
    parser.add_argument("--out-json", type=Path, default=Path("bench/results/quality_eval.json"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--n", type=int, default=6, help="reserved; current code samples 6")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_prompts = [json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()]
    picked = _pick_prompts(all_prompts, rng)

    print(f"Evaluating {len(picked)} prompts...")
    results: list[PromptResult] = []
    for i, p in enumerate(picked, 1):
        print(f"  [{i}/{len(picked)}] {p['prompt_id']} ({len(p['content'])} chars)")
        r = asyncio.run(evaluate(
            p["prompt_id"], p["content"], rng, args.max_tokens, not args.no_judge,
        ))
        results.append(r)
        if r.judge:
            print(f"    judge: winner={r.judge.winner}  swapped={r.judge.swapped_for_judging}")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_serialize(results), indent=2))
    _write_markdown(results, args.out_md)
    print(f"\nwrote {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
