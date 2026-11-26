"""Three rule-based detectors, each giving one of the five classes per meter and step.

All three share the same data-quality layer, which any AMI head-end has: a meter whose
reading did not arrive fresh is `comm_loss`. They differ in what they do with the rest.

1. wls      Baseline. WLS state estimation, chi-square test, then the largest normalised
            residual (LNR) names one meter as bad data -> `meter_fault`. It has no notion
            of a real load change or an attack.
2. peer     What Etrack did. Each customer's P is predicted from the 5 customers whose
            history it tracks best; a sustained deviation -> `meter_fault`.
3. balance  Peer check plus the energy balance at the trusted transformer meter.
            Deviation and balance holds -> `load_change` (the grid confirms it).
            Deviation and balance broken -> `meter_fault` if the meter reads low,
            `attack` if it reads high (meters rarely over-read; attackers can go either
            way, so downward attacks land in `meter_fault`).

The peer and balance models are fitted on 21 days of normal operation (the base
simulation, days 0-20, with fresh meter noise); none of the episodes or labels are used.
Thresholds are the 99.5th percentile of the same statistic on that history.

Readings that did not arrive fresh are replaced, for the estimator and the balance, by a
pseudo-measurement: the last fresh value with a wide std (30 % for P/Q, 2 % for V), as an
operator would do to keep the grid observable.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import estimation as est
import grid
import measurements as meas

WINDOW = 8          # steps (2 hours) for the rolling mean of peer and balance scores
N_PEERS = 5
QUANTILE = 0.995    # threshold = this quantile of |rolling score| on normal history
HISTORY_DAYS = 21


def __getattr__(name):  # OUT_DIR follows the active run (grid.use_run)
    if name == "OUT_DIR":
        return grid.DATA_DIR / "detectors"
    raise AttributeError(name)


# --- shared helpers -------------------------------------------------------------------

def meter_columns(defs: pd.DataFrame, meas_type: str = "p") -> pd.Series:
    """meter -> measurement id of its `meas_type` reading (customer meters only)."""
    sel = defs[(defs.group == "customer_pq") & (defs.meas_type == meas_type)]
    return pd.Series(sel.index, index=sel.meter)


def fresh_by_meter(defs: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    return fresh.T.groupby(defs.meter).all().T


def operator_view(defs: pd.DataFrame, readings: pd.DataFrame, fresh: pd.DataFrame):
    """Values and std the operator would feed to the estimator (pseudo for non-fresh)."""
    vals = readings.where(fresh).groupby(level="episode").ffill()
    std = meas.std_devs(defs, vals)
    is_v = (defs.meas_type == "v").to_numpy()
    wide = np.where(is_v, 0.02, np.maximum(0.3 * vals.abs().to_numpy(), 2e-4))
    std = std.where(fresh, pd.DataFrame(wide, index=std.index, columns=std.columns))
    return vals, std


def history(net, defs, seed: int = 99) -> pd.DataFrame:
    """Noisy readings for the normal history (days 0-20 of the base simulation)."""
    base = grid.load_results(grid.DATA_DIR / "pf")
    true = meas.true_values(defs, base).iloc[: HISTORY_DAYS * grid.STEPS_PER_DAY]
    return meas.add_noise(true, meas.std_devs(defs, true), seed=seed)


def _rolling_threshold(score: pd.DataFrame | pd.Series):
    return score.rolling(WINDOW).mean().abs().quantile(QUANTILE)


# --- 1. WLS baseline --------------------------------------------------------------------

_W = {}


def _init_worker(run, defs, vals, std):
    # Worker processes start fresh: silence pandapower's pandas warnings here too,
    # or they flood the notebook output.
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    grid.use_run(run)
    _W.update(net=grid.load_net(), defs=defs, vals=vals, std=std)


def _wls_episode(ep: int) -> pd.DataFrame:
    net, defs, vals, std = _W["net"], _W["defs"], _W["vals"].loc[ep], _W["std"].loc[ep]
    rows = []
    for step in vals.index:
        res = est.residual_tests(net, defs, vals.loc[step], std.loc[step])
        row = {"episode": ep, "step": step, "ok": res["ok"], "J": res["J"], "flag": res["flag"]}
        row.update({f"rn_{i}": v for i, v in res["rn"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def run_wls_baseline(data: dict, workers: int = 8) -> pd.DataFrame:
    """Chi-square and normalised residuals for every (episode, step). ~5 min on 8 cores."""
    defs = data["defs"]
    vals, std = operator_view(defs, data["readings"], data["fresh"])
    eps = list(data["episodes"].index)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(grid.RUN, defs, vals, std)) as ex:
        parts = list(ex.map(_wls_episode, eps, chunksize=4))
    return pd.concat(parts).set_index(["episode", "step"]).sort_index()


def wls_predictions(data: dict, wls: pd.DataFrame, rn_threshold: float = 3.0) -> pd.DataFrame:
    """Baseline classes. Chi-square alarm + LNR over threshold -> that meter is bad data."""
    defs, labels = data["defs"], data["labels"]
    pred = pd.DataFrame("normal", index=labels.index, columns=labels.columns)
    rn = wls.filter(like="rn_")
    rn.columns = [int(c[3:]) for c in rn.columns]
    worst = rn.idxmax(axis=1)
    hit = wls.flag & (rn.max(axis=1) > rn_threshold)
    for (ep, step), mid in worst[hit].astype(int).items():
        pred.at[(ep, step), defs.meter[mid]] = "meter_fault"
    pred[~fresh_by_meter(defs, data["fresh"]).reindex(columns=pred.columns, fill_value=True)] = "comm_loss"
    return pred


# --- 2. Peer check ------------------------------------------------------------------------

def fit_peers(hist: pd.DataFrame, defs: pd.DataFrame) -> dict:
    """For each customer meter: 5 best-correlated peers, a linear fit, residual std, threshold."""
    pcol = meter_columns(defs)
    p = hist[pcol.to_numpy()]
    p.columns = pcol.index
    corr = p.corr()
    model = {}
    for m in p.columns:
        peers = corr[m].drop(m).nlargest(N_PEERS).index.tolist()
        x = np.column_stack([np.ones(len(p)), p[peers].to_numpy()])
        coef, *_ = np.linalg.lstsq(x, p[m].to_numpy(), rcond=None)
        resid = p[m].to_numpy() - x @ coef
        sigma = resid.std()
        thr = _rolling_threshold(pd.Series(resid / sigma))
        model[m] = {"peers": peers, "coef": coef, "sigma": sigma, "threshold": thr}
    return model


def peer_scores(data: dict, model: dict):
    """Rolling peer z-score per meter and step, and whether the peers were all available."""
    defs = data["defs"]
    vals, _ = operator_view(defs, data["readings"], data["fresh"])
    pcol = meter_columns(defs)
    p = vals[pcol.to_numpy()]
    p.columns = pcol.index
    fr = fresh_by_meter(defs, data["fresh"])
    z = pd.DataFrame(index=p.index, columns=p.columns, dtype=float)
    covered = pd.DataFrame(True, index=p.index, columns=p.columns)
    for m, mod in model.items():
        pred = mod["coef"][0] + p[mod["peers"]].to_numpy() @ mod["coef"][1:]
        z[m] = (p[m] - pred) / mod["sigma"]
        covered[m] = fr[mod["peers"]].all(axis=1)
    z = z.where(covered)
    zbar = z.groupby(level="episode").transform(lambda s: s.rolling(WINDOW, min_periods=WINDOW).mean())
    thr = pd.Series({m: mod["threshold"] for m, mod in model.items()})
    return zbar, zbar.abs() > thr, covered


def peer_predictions(data: dict, flag: pd.DataFrame) -> pd.DataFrame:
    labels = data["labels"]
    pred = pd.DataFrame("normal", index=labels.index, columns=labels.columns)
    pred[flag.reindex(columns=pred.columns, fill_value=False)] = "meter_fault"
    pred[~fresh_by_meter(data["defs"], data["fresh"]).reindex(columns=pred.columns, fill_value=True)] = "comm_loss"
    return pred


# --- 3. Energy balance ----------------------------------------------------------------------

def _balance_terms(defs, vals):
    tp = defs.index[(defs.meter == "TRAFO") & (defs.meas_type == "p")][0]
    tq = defs.index[(defs.meter == "TRAFO") & (defs.meas_type == "q")][0]
    gap = -vals[tp] - vals[meter_columns(defs).to_numpy()].sum(axis=1)  # losses + error
    x = np.column_stack([np.ones(len(vals)), vals[tp] ** 2, vals[tq] ** 2])
    return gap.to_numpy(), x


def fit_balance(hist: pd.DataFrame, defs: pd.DataFrame) -> dict:
    """Loss model gap = a + b P_trafo^2 + c Q_trafo^2, residual std and threshold."""
    gap, x = _balance_terms(defs, hist)
    coef, *_ = np.linalg.lstsq(x, gap, rcond=None)
    resid = gap - x @ coef
    sigma = resid.std()
    return {"coef": coef, "sigma": sigma, "threshold": _rolling_threshold(pd.Series(resid / sigma))}


def balance_scores(data: dict, model: dict):
    """Rolling normalised balance mismatch per (episode, step); positive = meters under-report."""
    defs = data["defs"]
    vals, _ = operator_view(defs, data["readings"], data["fresh"])
    gap, x = _balance_terms(defs, vals)
    e = pd.Series((gap - x @ model["coef"]) / model["sigma"], index=vals.index)
    ebar = e.groupby(level="episode").transform(lambda s: s.rolling(WINDOW, min_periods=WINDOW).mean())
    return ebar, ebar.abs() > model["threshold"]


def balance_predictions(data: dict, peer_flag, peer_z, bal_flag) -> pd.DataFrame:
    labels = data["labels"]
    pred = pd.DataFrame("normal", index=labels.index, columns=labels.columns)
    flag = peer_flag.reindex(columns=pred.columns, fill_value=False)
    z = peer_z.reindex(columns=pred.columns)
    broken = pd.DataFrame(np.repeat(bal_flag.to_numpy()[:, None], pred.shape[1], axis=1),
                          index=pred.index, columns=pred.columns)
    pred[flag & ~broken] = "load_change"
    pred[flag & broken & (z < 0)] = "meter_fault"
    pred[flag & broken & (z > 0)] = "attack"
    pred[~fresh_by_meter(data["defs"], data["fresh"]).reindex(columns=pred.columns, fill_value=True)] = "comm_loss"
    return pred


# --- run everything ----------------------------------------------------------------------------

def run_all(data: dict, net=None, wls: pd.DataFrame | None = None) -> dict:
    """Fit on history, score every episode, return predictions and intermediate scores."""
    net = net if net is not None else grid.load_net()
    defs = data["defs"]
    hist = history(net, defs)
    peer_model, bal_model = fit_peers(hist, defs), fit_balance(hist, defs)
    peer_z, peer_flag, covered = peer_scores(data, peer_model)
    bal_e, bal_flag = balance_scores(data, bal_model)
    if wls is None:
        wls = run_wls_baseline(data)
    return {
        "pred": {"wls": wls_predictions(data, wls),
                 "peer": peer_predictions(data, peer_flag),
                 "balance": balance_predictions(data, peer_flag, peer_z, bal_flag)},
        "wls": wls, "peer_z": peer_z, "peer_flag": peer_flag, "peer_covered": covered,
        "bal_e": bal_e, "bal_flag": bal_flag, "peer_model": peer_model, "bal_model": bal_model,
    }


if __name__ == "__main__":
    import time
    import scenarios as sc
    t = time.time()
    data = sc.load()
    wls = run_wls_baseline(data)
    out_dir = grid.DATA_DIR / "detectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    wls.to_parquet(out_dir / "wls.parquet")
    print(f"WLS baseline: {len(wls)} steps, {int((~wls.ok).sum())} failed, "
          f"chi2 flag rate {wls.flag.mean():.3f}, {time.time() - t:.0f}s")
