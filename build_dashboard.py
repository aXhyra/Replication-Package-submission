"""
Build dashboard.html from the results the three notebooks write into analysis_out/.

    python3.10 build_dashboard.py

Run this script after the notebooks.
"""

from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RQ1_DIR = HERE / "analysis_out" / "rq1-healthiness"
RQ2_DIR = HERE / "analysis_out" / "rq2-prevention"
RQ3_DIR = HERE / "analysis_out" / "rq3-analysis"
TEMPLATE = HERE / "dashboard_template.html"
TARGET = HERE / "dashboard.html"

EVENT_START = 79200
TOP_METRICS = 8
TOP_TRAITS = 6
PLOT_STRIDE_FALLBACK = 120

COMBO_PARTS = {
    "bank_tellers_drivers": ["bank_tellers", "drivers"],
    "students_drivers": ["students", "drivers"],
    "managers_drivers": ["managers", "drivers"],
}

def clean(v, nd: int = 4):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, nd)

def safe_mean(series, nd=4):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return clean(s.mean(), nd) if len(s) else None

def safe_median(series, nd=0):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return clean(s.median(), nd) if len(s) else None


NOTEBOOK_OF = {RQ1_DIR: "rq1-healthiness.ipynb", RQ2_DIR: "rq2-prevention.ipynb", RQ3_DIR: "rq3-analysis.ipynb"}


def read(name: str, folder: Path = RQ3_DIR) -> pd.DataFrame:
    path = folder / name
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(HERE)}. Run {NOTEBOOK_OF[folder]} first.")
    return pd.read_csv(path)

def build_payload() -> dict:
    manifest_path = RQ3_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path.relative_to(HERE)}. Run rq3-analysis.ipynb first.")
    man = json.loads(manifest_path.read_text())

    dash = read("dashboard.csv")
    spikes = read("spikes.csv")
    decomp = read("spike_decomposition.csv")
    profile = read("spike_metric_profile.csv")
    attribution = read("attribution.csv")
    sig_scen = read("signature_scenario.csv")
    sig_pers = read("signature_personality.csv")
    add_scen = read("additivity_scenarios.csv")
    add_pers = read("additivity_personalities.csv")
    families = read("causes_by_family.csv")
    series = read("index_series.csv")
    windows = read("event_window_levels.csv")
    detail = read("failure_detail.csv")
    bl_cells = read("verdict_cells.csv")
    bl_fail = read("failing_spike_detail.csv")
    bl_prof = read("failing_metric_profile.csv")
    bl_sum = read("threshold_summary.csv")
    prior = read("prior_verdicts.csv", RQ1_DIR)
    conditions = read("failure_conditions.csv") if (RQ3_DIR / "failure_conditions.csv").exists() else pd.DataFrame()
    tolerance = read("failure_tolerance.csv") if (RQ3_DIR / "failure_tolerance.csv").exists() else pd.DataFrame()

    def conditions_of(p: str) -> list[dict]:
        if conditions.empty:
            return []
        return [{"m": r.metric, "f": r.family, "dir": r.direction, "sep": clean(r.separation, 3),
                 "gapZ": clean(r.gap_z, 2), "cut": clean(r.cut, 3), "acc": clean(r.accuracy, 3),
                 "medF": clean(r.median_fail, 3), "medP": clean(r.median_pass, 3),
                 "othersSep": clean(r.others_separation, 3), "specificity": clean(r.specificity, 3),
                 "specific": bool(r.specific), "nFail": int(r.n_fail), "nPass": int(r.n_pass)}
                for r in conditions[conditions.personality == p].itertuples()]

    scenarios = man["scenarios"]
    personalities = man["personalities"]

    # Spike episode of the reported failure timestamp
    dec_by_id = decomp.set_index("spike_id") if "spike_id" in decomp.columns else pd.DataFrame()
    fail_spike: dict[str, dict] = {}
    for _, s in spikes[spikes.contains_failure].iterrows():
        row = {"start": clean(s.start_ts, 0), "end": clean(s.end_ts, 0),
               "peak": clean(s.peak, 3), "mean": clean(s["mean"], 3),   # s.mean is Series.mean
               "dur": clean(s.duration_min, 1), "pos": s.position}
        if s.spike_id in dec_by_id.index:
            r = dec_by_id.loc[s.spike_id]
            row.update({"ns": clean(r.d_not_served), "ca": clean(r.d_cancel),
                        "de": clean(r.d_demand), "rise": clean(r.fr_rise),
                        "share_ns": clean(r.share_not_served, 3), "driver": r.dominant_driver,
                        "nsPre": clean(r.not_served_pre, 3), "nsIn": clean(r.not_served_spike, 3),
                        "caPre": clean(r.cancel_pre, 3), "caIn": clean(r.cancel_spike, 3)})
        fail_spike[f"{s.scenario}|{s.personality}"] = row

    # Consistency and resilience behind each verdict
    det = {f"{r.scenario}|{r.personality}": r for r in detail.itertuples()}
    for key, row in fail_spike.items():
        d = det.get(key)
        if d is not None:
            row.update({"conPeak": clean(d.consistency_peak, 4),
                        "resRun": clean(d.resilience_ticks, 0)})

    # Metrics most disturbed inside each cell's spikes
    index_own = set(man.get("index_components", [])) | {"failed_requests"}
    prof = profile[~profile.metric.isin(index_own)].copy()
    prof["_abs"] = prof.dev_vs_own.abs()
    affected: dict[str, list] = {}
    for (scen, pers), g in prof.sort_values("_abs", ascending=False).groupby(["scenario", "personality"]):
        affected[f"{scen}|{pers}"] = [
            {"m": r.metric, "f": r.family, "dev": clean(r.dev_vs_own, 2),
             "ref": clean(r.dev_vs_reference, 2),
             "lvl": clean(r.spike_level, 3), "base": clean(r.own_baseline_level, 3)}
            for r in g.head(TOP_METRICS).itertuples()
        ]

    # Index trace
    traces: dict[str, dict] = {}
    for (scen, pers), g in series.sort_values("timestamp").groupby(["scenario", "personality"]):
        ts = g.timestamp.to_numpy()
        traces[f"{scen}|{pers}"] = {
            "t0": int(ts[0]),
            "dt": int(ts[1] - ts[0]) if len(ts) > 1 else PLOT_STRIDE_FALLBACK,
            # index x1000, rounded: three decimals is finer than the plot can resolve
            "v": [int(round(x * 1000)) for x in g.failed_requests.to_numpy()],
        }

    # Levels of the two failure metrics over the event window (rides_not_served and passengers_cancel)
    win: dict[str, dict] = {}
    wsub = windows[windows.metric.isin(["rides_not_served", "passengers_cancel"])]
    for (scen, pers), g in wsub.groupby(["scenario", "personality"]):
        g = g.set_index("metric")
        win[f"{scen}|{pers}"] = {
            "nsPre": clean(g.at["rides_not_served", "pre"], 3),
            "nsIn": clean(g.at["rides_not_served", "during"], 3),
            "caPre": clean(g.at["passengers_cancel", "pre"], 3),
            "caIn": clean(g.at["passengers_cancel", "during"], 3),
        }

    def g(row, name, nd=4):
        return clean(getattr(row, name, None), nd)

    fail_by = {}
    for r in bl_fail.itertuples():
        fail_by[f"{r.scenario}|{r.personality}"] = {
            "start": clean(r.start_ts, 0), "end": clean(r.end_ts, 0),
            "dur": clean(r.duration_min, 1), "peak": clean(r.peak, 3),
            "mean": clean(r.mean, 3), "pos": r.position,
            "ns": g(r, "d_not_served"), "ca": g(r, "d_cancel"),
            "rise": g(r, "fr_rise"), "share_ns": g(r, "share_not_served", 3),
            "nsPre": g(r, "not_served_pre", 3), "nsIn": g(r, "not_served_spike", 3),
            "caPre": g(r, "cancel_pre", 3), "caIn": g(r, "cancel_spike", 3),
            "conPeak": clean(r.consistency_peak, 4), "resRun": clean(r.resilience_ticks, 0),
        }

    prof_b = bl_prof.copy()
    prof_b["_abs"] = prof_b.dev_vs_own.abs()
    aff_by = {}
    for (scen, pers), grp in (prof_b.sort_values("_abs", ascending=False)
                              .groupby(["scenario", "personality"])):
        aff_by[f"{scen}|{pers}"] = [
            {"m": r.metric, "f": r.family, "dev": clean(r.dev_vs_own, 2),
             "ref": clean(r.dev_vs_reference, 2),
             "lvl": clean(r.spike_level, 3), "base": clean(r.own_baseline_level, 3)}
            for r in grp.head(TOP_METRICS).itertuples()]
    per_run: dict[str, dict] = {}
    for r in bl_cells.itertuples():
        key = f"{r.scenario}|{r.personality}"
        per_run[key] = {
            "failure": bool(r.failure),
            "failure_timestamp": clean(r.failure_timestamp, 0),
            "n_spikes": int(r.n_spikes),
            "spike_minutes": clean(r.spike_minutes, 1),
            "spike_minutes_in_event": clean(r.spike_minutes_in_event, 1),
            "d_not_served": g(r, "d_not_served"), "d_cancel": g(r, "d_cancel"),
            "share_not_served": g(r, "share_not_served", 3),
            "failSpike": fail_by.get(key),
            "affected": aff_by.get(key, []),
        }

    def agg_by(group_col: str, listed: str) -> dict:
        return {name: {"nFail": int(sub.failure.sum()), "n": len(sub),
                       "failed": sub[sub.failure][listed].tolist(),
                       "spikeMin": clean(sub.spike_minutes.mean(), 1)}
                for name, sub in bl_cells.groupby(group_col)}

    scen_verdicts = agg_by("scenario", "personality")
    pers_verdicts = agg_by("personality", "scenario")
    cells = []
    for _, r in dash.iterrows():
        cell = {k: clean(r[k]) for k in dash.columns}
        key = f"{r.scenario}|{r.personality}"
        cell["failSpike"] = fail_spike.get(key)
        cell["affected"] = affected.get(key, [])
        cell["trace"] = traces.get(key)
        cell["window"] = win.get(key)
        cell["b"] = per_run.get(key, {})
        cells.append(cell)
    attr = attribution[~attribution.metric.isin(index_own)]

    def top_by(frame, col):
        t = (frame.groupby(["metric", "family"])[col].mean().reset_index()
             .assign(_a=lambda x: x[col].abs())
             .sort_values("_a", ascending=False).head(TOP_METRICS))
        return [{"m": r.metric, "f": r.family, "z": clean(getattr(r, col), 1)} for r in t.itertuples()]

    def traits_of(frame, key, value):
        t = frame[(frame[key] == value) & frame.is_trait].sort_values("abs_mean_z", ascending=False)
        return [{"m": r.metric, "z": clean(r.mean_z, 1), "d": r.direction, "f": r.family}
                for r in t.head(TOP_TRAITS).itertuples()]

    # Event-wise information
    scen_info = {}
    for s in scenarios:
        sub = dash[dash.scenario == s]
        fam = (families[families.scenario == s].groupby("family")["scenario_effect"]
               .mean().sort_values(ascending=False).head(6))
        window = man["event_windows"][s]
        info = {
            "components": (s.split("-") if s != "normal" else []),
            "mechanism": sub.mechanism.iloc[0],
            "window": [window[0], window[1]],
            "durationMin": (window[1] - window[0]) // 60,
            "nFail": int(sub.failure.eq(True).sum()),
            "n": len(sub),
            "failedBy": sub[sub.failure.eq(True)].personality.tolist(),
            "meanIndex": safe_mean(sub.fr_mean_event, 3),
            "meanIndexRange": [clean(sub.fr_mean_event.min(), 3), clean(sub.fr_mean_event.max(), 3)],
            "peak": clean(sub.fr_peak.max(), 3),
            "spikeMin": safe_mean(sub.spike_minutes, 1),
            "effect": clean(sub.fr_scenario_effect.iloc[0], 4),
            "medianFailTs": safe_median(sub.failure_timestamp),
            "topMetrics": top_by(attr[attr.scenario == s], "scenario_effect_z"),
            "families": [{"f": i, "v": clean(v, 2)} for i, v in fam.items()],
            "traits": traits_of(sig_scen, "scenario", s),
        }
        if len(info["components"]) > 1:
            fa = add_scen[(add_scen.scenario == s) & (add_scen.metric == "failed_requests")]
            info["additivity"] = {
                "expected": safe_mean(fa.expected), "observed": safe_mean(fa.observed),
                "residual": safe_mean(fa.residual),
                "superAdditive": int((fa.residual > 0).sum()), "n": len(fa),
            }
        info["b"] = scen_verdicts.get(s, {})
        scen_info[s] = info

    # Personality-wise information
    pers_info = {}
    for p in personalities:
        sub = dash[dash.personality == p]
        fam = (families[families.personality == p].groupby("family")["personality_effect"]
               .mean().sort_values(ascending=False).head(6))
        info = {
            "nFail": int(sub.failure.eq(True).sum()),
            "n": len(sub),
            "failedIn": sub[sub.failure.eq(True)].scenario.tolist(),
            "meanIndex": safe_mean(sub.fr_mean_event, 3),
            "peak": clean(sub.fr_peak.max(), 3),
            "spikeMin": safe_mean(sub.spike_minutes, 1),
            "effect": clean(sub.fr_personality_effect.iloc[0], 4),
            "topMetrics": top_by(attr[attr.personality == p], "personality_effect_z"),
            "families": [{"f": i, "v": clean(v, 2)} for i, v in fam.items()],
            "traits": traits_of(sig_pers, "personality", p),
            "conditions": conditions_of(p),
        }
        if p in COMBO_PARTS:
            fa = add_pers[(add_pers.personality == p) & (add_pers.metric == "failed_requests")]
            info["parts"] = COMBO_PARTS[p]
            info["additivity"] = {
                "expected": safe_mean(fa.expected), "observed": safe_mean(fa.observed),
                "residual": safe_mean(fa.residual),
                "superAdditive": int((fa.residual > 0).sum()), "n": len(fa),
            }
        info["b"] = pers_verdicts.get(p, {})
        pers_info[p] = info

    failure = dash.failure.eq(True)
    ts = pd.to_numeric(dash.failure_timestamp, errors="coerce")

    meta = {k: man[k] for k in ("scenarios", "personalities", "threshold_baseline",
                                "spike_threshold", "event_windows", "preprocessing",
                                "reference_config", "index_sources")}
    meta.update({
        "compound": man["compound_scenarios"],
        "thresholds": {k: clean(v, 6) for k, v in man["thresholds"].items()},
        "nFail": int(failure.sum()),
        "nPass": int((~failure).sum()),
        "nSpikes": int(len(spikes)),
        "earlyFailureCutoff": EVENT_START,
        "nEarly": int((ts < EVENT_START).sum()),
        "threshold": {
            "label": str(bl_sum.iloc[0].label),
            "verdicts": str(getattr(bl_sum.iloc[0], "verdicts", "replayed")),
            "p99_h": clean(bl_sum.iloc[0].p99_h, 6), "p99_c": clean(bl_sum.iloc[0].p99_c, 6),
            "p99_r": int(bl_sum.iloc[0].p99_r), "failures": int(bl_sum.iloc[0].failures),
            "runs": int(bl_sum.iloc[0].runs), "spikes": int(bl_sum.iloc[0].spikes)},
        "paper": {r.scenario: ({"failure": bool(r.failure)} if bool(r.in_prior) else None)
                  for r in prior.itertuples()},
        "tolerance": [
            {"m": m, "f": g.family.iloc[0], "meanSep": clean(g.mean_separation.iloc[0], 3),
             "cuts": {r.personality: clean(r.cut, 3) for r in g.itertuples()},
             "dirs": {r.personality: r.direction for r in g.itertuples()},
             "seps": {r.personality: clean(r.separation, 3) for r in g.itertuples()}}
            for m, g in (tolerance.groupby("metric", sort=False) if not tolerance.empty else [])
        ][:12],
        "paperOnly": {
            "single": [r.scenario for r in prior.itertuples()
                       if getattr(r, "prior_only_group", "") == "single"],
            "compound": [r.scenario for r in prior.itertuples()
                         if getattr(r, "prior_only_group", "") == "compound"],
        },
    })

    return {"meta": meta, "cells": cells, "scenarios": scen_info, "personalities": pers_info,
            "prevention": prevention_payload(index_own)}

# Runs executed with a prevention strategy
def prevention_payload(index_own: set[str] | None = None) -> dict:
    if not RQ2_DIR.exists():
        return {}
    out: dict[str, dict] = {}
    for path in sorted(RQ2_DIR.glob("*/cells.json")):
        payload = json.loads(path.read_text())
        effects_path = path.with_name("effects.json")
        effects = json.loads(effects_path.read_text()) if effects_path.exists() else {}
        cells = {}
        dropped = 0
        for c in payload["cells"]:
            if c.get("base_failure") is not True:
                dropped += 1
                continue
            if index_own:
                c["affected"] = [m for m in c.get("affected", [])
                                 if m["metric"] not in index_own]
            c.pop("window_levels", None)
            cells[f"{c['scenario']}|{c['personality']}"] = c
        shown = list(cells.values())
        out[payload["strategy"]] = {
            "label": payload.get("label", payload["strategy"]),
            "threshold": payload.get("threshold"),
            "cells": cells,
            "counts": {k: sum(1 for c in shown if c.get("outcome") == k)
                       for k in ("prevented", "still fails")},
            "dropped_passing": dropped,
            "effects": effects,
        }
    return out

def main() -> int:
    if not TEMPLATE.exists():
        raise SystemExit(f"Missing {TEMPLATE.name}.")

    payload = build_payload()
    data = json.dumps(payload, separators=(",", ":"))
    if "</script" in data.lower():
        raise SystemExit("Broken page: payload contains a literal </script>.")
    template = TEMPLATE.read_text()
    if "__DATA__" not in template:
        raise SystemExit(f"{TEMPLATE.name} has no __DATA__ placeholder.")
    TARGET.write_text(template.replace("__DATA__", data, 1))

    m = payload["meta"]
    print(f"Wrote {TARGET.relative_to(HERE)}  ({TARGET.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(payload['cells'])} runs · {len(payload['scenarios'])} events × {len(payload['personalities'])} personalities")
    for name, variant in payload.get("prevention", {}).items():
        counts = {k: v for k, v in variant["counts"].items() if v}
        extra = (f" (+{variant['dropped_passing']} runs hidden: they pass without prevention)" if variant.get("dropped_passing") else "")
        print(f"  prevention · {variant['label']}: {len(variant['cells'])} runs {counts}{extra}")
    t = m["threshold"]
    print(f"  {t['label']}: {t['failures']} fail / {t['runs'] - t['failures']} pass · {t['spikes']} spike episodes · verdicts {t['verdicts']}")
    print(f"  index source: {m['index_sources']}")

    return 0

if __name__ == "__main__":
    sys.exit(main())