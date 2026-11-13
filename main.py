#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster for make.fun competition
roblox-up-and-coming-forecast-01.

Reads the chart's history (/input/history.ndjson + task.json) and writes one
forecast (/output/forecast.json): per-place probabilities for each game, and
the top 10 in order. Scored against every hourly read of the 72h window after
the deadline, by log score (mean over reads of sum_k ln(p_k)).

Method: empirical rank-transition kernel. The chart is sorted by user
GROWTH, not by player count, so a game's player count does not determine its
place (a game with 59,686 playing can sit at #5 while one with 8,057 leads).
What matters is the rank itself, so the forecast models the rank directly:
for each game, its future rank is a normal distribution centred on its
current rank plus its rank band's drift, with a spread that grows with the
horizon (calibrated from the history's own consecutive-read rank changes),
and each place's probability is that distribution's mass on the place.
Averaged over the 72 hourly horizons of the window.

A per-place newcomer reserve is left unnamed and grows with the horizon:
games born after the history ends take a growing share of the top 10, and no
history can name them, so part of every place must stay open for them.

Standard library only, no network, deterministic: fully ordered iteration,
so two runs on the same history write byte-identical output.
"""

import json
import math
import os
from datetime import datetime, timezone


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


WINSOR_D = 5.0            # rank-change per hour clipped here (estimation)
DRIFT_SHRINK_K = 20.0     # pseudo-count shrinking a band's drift to 0
SIGMA_FLOOR = 0.3         # per-hour rank sd floor
SIGMA_CAP = 2.0           # per-hour rank sd cap
ABSENT_PENALTY = 15.0     # ranks added to a game off the last chart
ABSENT_WEIGHT = 0.3       # weight of a game off the last chart
# Newcomer reserve per place, growing with horizon h (hours since the cut):
#   q(h) = Q0 + (QMAX - Q0) * (1 - exp(-h / TAU_NEW_H))
NEWCOMER_Q0 = _f("MF_Q0", 0.10)
NEWCOMER_QMAX = _f("MF_QMAX", 0.30)
NEWCOMER_TAU_H = 24.0
MIN_P = 0.001             # smallest probability row emitted


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _band(rank):
    if rank <= 10:
        return 0
    if rank <= 30:
        return 1
    return 2


def build_forecast(history, places):
    """history: list of snapshot dicts (oldest first), each with 'at' and
    'up_and_coming' rows {rank, universe_id, playing}. Returns the forecast
    dict {"probabilities": [...], "top10": [...]}."""
    marks = [m for m in history if m.get("ok") is True
             and isinstance(m.get("up_and_coming"), list)]
    if not marks:
        raise SystemExit("the history holds no ok snapshot")
    marks.sort(key=lambda m: m["at"])
    n_marks = len(marks)
    times = [parse_time(m["at"]) for m in marks]

    # --- Calibrate rank dynamics from consecutive reads --------------------
    # Per-hour rank change, per rank band, and the overall spread.
    band_d, band_n = [0.0, 0.0, 0.0], [0, 0, 0]
    all_d = []
    for a, b in zip(range(n_marks - 1), range(1, n_marks)):
        dt_h = (times[b] - times[a]).total_seconds() / 3600.0
        if dt_h < 1.0 or dt_h > 24.0:
            continue
        ra = {int(r["universe_id"]): int(r["rank"]) for r in marks[a]["up_and_coming"]}
        rb = {int(r["universe_id"]): int(r["rank"]) for r in marks[b]["up_and_coming"]}
        for uid in ra:
            if uid in rb:
                d = (rb[uid] - ra[uid]) / dt_h
                d = min(max(d, -WINSOR_D), WINSOR_D)
                band = _band(ra[uid])
                band_d[band] += d
                band_n[band] += 1
                all_d.append(d)
    drifts = []
    for band in range(3):
        if band_n[band]:
            drifts.append(band_d[band] / (band_n[band] + DRIFT_SHRINK_K))
        else:
            drifts.append(0.0)
    if all_d:
        mean_d = sum(all_d) / len(all_d)
        var_d = sum((d - mean_d) ** 2 for d in all_d) / len(all_d)
        sigma = math.sqrt(var_d) if var_d > 0 else 0.5
    else:
        sigma = 0.5
    sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)

    # --- Universe and current ranks -----------------------------------------
    last_rows = sorted(marks[-1]["up_and_coming"], key=lambda r: r["rank"])
    if len(last_rows) < places:
        raise SystemExit("the last snapshot ranks fewer games than the places")
    n_chart = len(last_rows)
    cur = {}  # uid -> (rank0, weight)
    for r in last_rows:
        cur[int(r["universe_id"])] = (float(r["rank"]), 1.0)
    # Recent absentees: games on the chart in the last 6 snapshots but not now.
    last_seen = {}
    for i, m in enumerate(marks):
        for r in m["up_and_coming"]:
            last_seen[int(r["universe_id"])] = (i, int(r["rank"]))
    for uid, (i, rank) in sorted(last_seen.items()):
        if uid not in cur and i >= n_marks - 6:
            cur[uid] = (float(rank) + ABSENT_PENALTY, ABSENT_WEIGHT)

    # --- Per-place probabilities, averaged over the 72 hourly horizons -----
    prob_sum = {k: {} for k in range(1, places + 1)}
    horizons = list(range(1, 73))
    for h in horizons:
        q = NEWCOMER_Q0 + (NEWCOMER_QMAX - NEWCOMER_Q0) * (
            1.0 - math.exp(-float(h) / NEWCOMER_TAU_H))
        sd = sigma * math.sqrt(float(h))
        for uid, (r0, w) in cur.items():
            m = r0 + drifts[_band(int(r0))] * h
            # mass on each place k: Phi((k+0.5-m)/sd) - Phi((k-0.5-m)/sd)
            for k in range(1, places + 1):
                p = _ncdf((k + 0.5 - m) / sd) - _ncdf((k - 0.5 - m) / sd)
                if p > 0.0:
                    prob_sum[k][uid] = prob_sum[k].get(uid, 0.0) + p * w * (1.0 - q)
    n_h = len(horizons)
    probabilities = []
    for k in range(1, places + 1):
        for uid, p in sorted(prob_sum[k].items(), key=lambda kv: (-kv[1], kv[0])):
            p /= n_h
            if p >= MIN_P:
                probabilities.append({"universe_id": uid, "place": k, "p": round(p, 6)})

    # top10: greedy best game per place, distinct.
    last_order = [int(r["universe_id"]) for r in last_rows]
    top10 = []
    used = set()
    for k in range(1, places + 1):
        best = None
        for row in probabilities:
            if row["place"] == k and row["universe_id"] not in used:
                best = row["universe_id"]
                break
        if best is None:
            for uid in last_order:
                if uid not in used:
                    best = uid
                    break
        top10.append(best)
        used.add(best)

    return {"probabilities": probabilities, "top10": top10}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(in_dir, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    history = []
    with open(os.path.join(in_dir, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            history.append(json.loads(line))
    forecast = build_forecast(history, places)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, sort_keys=True, separators=(",", ":")))
    print("wrote %s (%d probability rows)" % (out_path, len(forecast["probabilities"])))
    print("top10:", forecast["top10"])


if __name__ == "__main__":
    main()
