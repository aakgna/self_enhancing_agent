"""Tune SIMILARITY_FLOOR against the hand-labeled request set.

Offline mode (default, no API key needed): sweeps the floor as a pure gap
detector — predict "none" iff no candidate clears the floor — and reports how
each floor separates covered vs uncovered requests.

Live mode (--live, needs ANTHROPIC_API_KEY): runs the full two-stage
classifier (floor + Haiku) per labeled request and reports end-to-end routing
accuracy — this is the Phase 2 exit criterion.

    uv run python -m eval.tune_threshold [--live]
"""

import argparse
import json
from pathlib import Path

from orchestrator import classifier, registry

LABELED = json.loads((Path(__file__).parent / "labeled_requests.json").read_text())
USER = "_tuning"


def offline_sweep() -> None:
    scored = []
    for item in LABELED:
        hits = registry.search(item["request"], USER, top_k=5, type_filter="agent")
        top_sim = hits[0].similarity if hits else 0.0
        scored.append((item, top_sim, hits[0].manifest.id if hits else None))

    print(f"{'floor':>6} {'gap_prec':>9} {'gap_rec':>8} {'covered_top1':>13} {'overall':>8}")
    for floor in [round(0.30 + 0.05 * i, 2) for i in range(9)]:
        tp = fp = fn = 0  # gap detection: positive = "none"
        covered_top1 = covered_total = correct_all = 0
        for item, top_sim, top_id in scored:
            is_gap = item["expected"] == "none"
            predicted_gap = top_sim < floor
            if predicted_gap and is_gap:
                tp += 1
            elif predicted_gap and not is_gap:
                fp += 1
            elif not predicted_gap and is_gap:
                fn += 1
            if not is_gap:
                covered_total += 1
                if not predicted_gap and top_id == item["expected"]:
                    covered_top1 += 1
            correct = (predicted_gap and is_gap) or (
                not predicted_gap and top_id == item["expected"] and not is_gap
            )
            correct_all += correct
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print(
            f"{floor:>6} {precision:>9.0%} {recall:>8.0%} "
            f"{covered_top1}/{covered_total:>10} {correct_all / len(scored):>8.0%}"
        )
    print(
        "\nNote: the floor alone is a coarse pre-filter; the Haiku stage makes the"
        "\nfinal call on everything above the floor. Run with --live for the real"
        "\nend-to-end number (Phase 2 exit criterion)."
    )


def live_eval() -> None:
    correct = 0
    errors = []
    for item in LABELED:
        hits = registry.search(item["request"], USER, top_k=5, type_filter="agent")
        result = classifier.classify(item["request"], hits)
        predicted = "none" if result.decision == classifier.NO_MATCH else result.decision
        ok = predicted == item["expected"]
        correct += ok
        if not ok:
            errors.append((item["request"], item["expected"], predicted))
        print(f"{'OK ' if ok else 'MISS'} {item['request'][:55]:57} -> {predicted}")
    print(f"\nend-to-end accuracy: {correct}/{len(LABELED)} = {correct / len(LABELED):.0%}")
    if errors:
        print("misses:")
        for request, expected, predicted in errors:
            print(f"  {request!r}: expected {expected}, got {predicted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    registry.init_registry()
    registry.backfill()
    if args.live:
        live_eval()
    else:
        offline_sweep()
