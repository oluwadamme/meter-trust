"""Gradient-boosted classifier on per-meter, per-step features, split by time.

Every feature uses only the current and past steps of the episode (no look-ahead), and
only what an operator has: the readings, whether they arrived fresh, the transformer
meter, and the outputs of the three rule-based detectors (WLS residuals, peer score,
energy balance). Rows are customer meters; the trusted transformer meter is not labelled.

Split by time, never at random (neighbouring steps of one episode are nearly identical
and would leak):
    fit         episodes inside days 0-14
    calibrate   episodes inside days 15-20: sets the alarm threshold
    test        episodes inside days 21-27
Trained with balanced class weights, the model flags far too many normal steps. So it
only raises an alarm when P(not normal) exceeds a threshold chosen on the calibration
days to give the same false-alarm budget as the rule-based detectors (0.6 % of normal
meter-steps); the alarm's class is the most likely non-normal class.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

import detectors as det
import grid

W = det.WINDOW  # 8 steps = 2 hours


def _roll(df: pd.DataFrame, fn: str) -> pd.DataFrame:
    """Rolling statistic over the previous W steps *within* each episode."""
    return df.groupby(level="episode").transform(lambda s: getattr(s.rolling(W, min_periods=2), fn)())


def build_features(data: dict, out: dict) -> pd.DataFrame:
    """One row per (episode, step, meter) for customer meters. Columns = features + label."""
    defs, E = data["defs"], data["episodes"]
    vals, _ = det.operator_view(defs, data["readings"], data["fresh"])
    pcol, qcol = det.meter_columns(defs, "p"), det.meter_columns(defs, "q")
    meters = list(pcol.index)
    p = vals[pcol.to_numpy()].set_axis(meters, axis=1) * 1e3       # kW
    q = vals[qcol.reindex(meters).to_numpy()].set_axis(meters, axis=1) * 1e3

    fresh = det.fresh_by_meter(defs, data["fresh"])[meters]
    gw = defs.drop_duplicates("meter").set_index("meter").gateway.reindex(meters)
    gw_missing = (~fresh).T.groupby(gw).transform("mean").T     # share of my gateway silent

    prev_mean = _roll(p, "mean").groupby(level="episode").shift(1)
    angle = np.degrees(np.arctan2(q, p.abs() + 1e-3))
    hist = det.history(None, defs)
    hist_angle = np.degrees(np.arctan2(hist[qcol.reindex(meters).to_numpy()].to_numpy(),
                                       np.abs(hist[pcol.to_numpy()].to_numpy()) + 1e-6))
    angle_dev = angle - np.median(hist_angle, axis=0)

    rn = out["wls"].filter(like="rn_")
    rn.columns = [int(c[3:]) for c in rn.columns]
    rn_meter = rn.T.groupby(defs.meter.reindex(rn.columns).to_numpy()).max().T.reindex(columns=meters)
    worst_meter = defs.meter.reindex(rn.idxmax(axis=1).to_numpy()).to_numpy()

    hour = ((E.base_start.reindex(p.index.get_level_values("episode")).to_numpy()
             + p.index.get_level_values("step").to_numpy()) % grid.STEPS_PER_DAY) / 4

    feats = {
        "p_kw": p,
        "change_vs_2h": (p - prev_mean) / (prev_mean.abs() + 0.1),
        "flatness_2h": _roll(p, "std"),
        "angle_dev_deg": angle_dev,
        "peer_z": out["peer_z"][meters],
        "peer_covered": out["peer_covered"][meters].astype(float),
        "fresh": fresh.astype(float),
        "not_fresh_2h": _roll((~fresh).astype(float), "sum"),
        "gateway_silent_share": gw_missing,
        "wls_rn_max": rn_meter,
        "wls_is_worst": pd.DataFrame(worst_meter[:, None] == np.array(meters)[None, :],
                                     index=p.index, columns=meters).astype(float),
    }
    feeder = {"balance_e": out["bal_e"], "wls_J": out["wls"].J, "hour": pd.Series(hour, index=p.index)}

    long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    long.index.names = ["episode", "step", "meter"]
    for k, s in feeder.items():
        long[k] = s.reindex(long.index.droplevel("meter")).to_numpy()
    long["label"] = data["labels"][meters].stack(future_stack=True).reindex(long.index).to_numpy()
    long["split"] = E.split.reindex(long.index.get_level_values("episode")).to_numpy()
    return long


FEATURES = ["p_kw", "change_vs_2h", "flatness_2h", "angle_dev_deg", "peer_z", "peer_covered",
            "fresh", "not_fresh_2h", "gateway_silent_share", "wls_rn_max", "wls_is_worst",
            "balance_e", "wls_J", "hour"]


FALSE_ALARM_BUDGET = 0.006
FIT_DAYS = det.HISTORY_DAYS  # the rule detectors learn from these days too, never beyond


def _part(feat: pd.DataFrame, episodes: pd.DataFrame) -> pd.Series:
    """fit / calibrate / test for every row, by where its episode's window lies in time."""
    start = episodes.base_start
    end = start + 96
    part = pd.Series("test", index=episodes.index)
    part[(episodes.split == "train") & (end <= FIT_DAYS * grid.STEPS_PER_DAY)] = "fit"
    part[(episodes.split == "train") & (start >= FIT_DAYS * grid.STEPS_PER_DAY)] = "calibrate"
    part[(episodes.split == "train") & (start < FIT_DAYS * grid.STEPS_PER_DAY)
         & (end > FIT_DAYS * grid.STEPS_PER_DAY)] = "unused"  # straddles the boundary
    return pd.Series(part.reindex(feat.index.get_level_values("episode")).to_numpy(), index=feat.index)


def train(feat: pd.DataFrame, episodes: pd.DataFrame, seed: int = 0):
    """Fit on days 0-14, set the alarm threshold on days 15-20. Returns (model, threshold)."""
    part = _part(feat, episodes)
    fit = feat[part == "fit"]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                           class_weight="balanced", random_state=seed)
    model.fit(fit[FEATURES], fit.label)
    cal = feat[(part == "calibrate") & (feat.label == "normal")]
    not_normal = 1 - model.predict_proba(cal[FEATURES])[:, list(model.classes_).index("normal")]
    threshold = float(np.quantile(not_normal, 1 - FALSE_ALARM_BUDGET))
    return model, threshold


def predict(model, threshold: float, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = np.array(model.classes_)
    i_norm = list(classes).index("normal")
    others = proba.copy()
    others[:, i_norm] = -1
    best = classes[others.argmax(axis=1)]
    return np.where(1 - proba[:, i_norm] > threshold, best, "normal")


def predict_frame(model, threshold: float, feat: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Predictions in the same (episode, step) x meter layout as the rule detectors."""
    pred = pd.Series(predict(model, threshold, feat[FEATURES]), index=feat.index).unstack("meter")
    pred = pred.reindex(index=labels.index.intersection(pred.index, sort=False))
    pred.insert(0, "TRAFO", "normal")
    return pred.reindex(columns=labels.columns, fill_value="normal")


def importance(model, feat: pd.DataFrame, n_normal: int = 20000, seed: int = 0) -> pd.Series:
    """Permutation importance (drop in macro-F1 of the raw model) on test rows: all events
    plus a sample of normal rows."""
    te = feat[feat.split == "test"]
    sample = pd.concat([te[te.label != "normal"],
                        te[te.label == "normal"].sample(n_normal, random_state=seed)])
    r = permutation_importance(model, sample[FEATURES], sample.label, scoring="f1_macro",
                               n_repeats=3, random_state=seed, n_jobs=-1)
    return pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)
