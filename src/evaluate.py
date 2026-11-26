"""Metrics and figures: confusion matrices, per-class precision/recall, detection delay."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scenarios import CLASSES

NAMES = {"wls": "1. WLS + chi-square + LNR", "peer": "2. Peer check",
         "balance": "3. Peer check + energy balance"}


def _customers(df: pd.DataFrame, episodes=None) -> pd.DataFrame:
    df = df.drop(columns="TRAFO", errors="ignore")
    return df if episodes is None else df.loc[list(episodes)]


def confusion(labels: pd.DataFrame, pred: pd.DataFrame, episodes=None) -> pd.DataFrame:
    """Meter-step confusion matrix over customer meters (rows: truth, columns: predicted)."""
    lab = _customers(labels, episodes).stack()
    prd = _customers(pred, episodes).stack()
    return pd.crosstab(lab.rename("true"), prd.rename("predicted")).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)


def per_class(cm: pd.DataFrame) -> pd.DataFrame:
    tp = np.diag(cm.to_numpy())
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = tp / cm.sum(axis=0).to_numpy()
        recall = tp / cm.sum(axis=1).to_numpy()
    return pd.DataFrame({"precision": precision, "recall": recall,
                         "support": cm.sum(axis=1).to_numpy()}, index=cm.index)


def detection(episodes: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Per event: was the right class ever given to an affected meter, and how fast?

    `delay` counts steps (15 min) from the event start to the first correct prediction;
    `alarm_delay` to the first prediction other than normal (any class).
    """
    rows = []
    for ep, e in episodes[episodes.cls != "normal"].iterrows():
        after = pred.loc[ep].loc[e.event_start:, e.affected.split(",")]
        correct = (after == e.cls).any(axis=1)
        alarm = (after != "normal").any(axis=1)
        rows.append({"episode": ep, "cls": e.cls, "subtype": e.subtype, "split": e.split,
                     "detected": correct.any(),
                     "delay": correct.idxmax() - e.event_start if correct.any() else np.nan,
                     "alarmed": alarm.any(),
                     "alarm_delay": alarm.idxmax() - e.event_start if alarm.any() else np.nan})
    return pd.DataFrame(rows).set_index("episode")


def detection_summary(det: pd.DataFrame) -> pd.DataFrame:
    g = det.groupby(["cls", "subtype"])
    return pd.DataFrame({"events": g.size(), "detected": g.detected.mean(),
                         "median delay (steps)": g.delay.median(),
                         "any alarm": g.alarmed.mean(),
                         "median alarm delay": g.alarm_delay.median()}).round(2)


def false_alarm_rate(labels: pd.DataFrame, pred: pd.DataFrame) -> float:
    """Share of truly normal customer meter-steps given any other class."""
    lab, prd = _customers(labels).stack(), _customers(pred).stack()
    normal = lab == "normal"
    return float((prd[normal] != "normal").mean())


# --- figures ---------------------------------------------------------------------------------

def plot_confusions(cms: dict, path=None):
    fig, axs = plt.subplots(1, len(cms), figsize=(5.2 * len(cms), 4.6))
    for ax, (name, cm) in zip(np.atleast_1d(axs), cms.items()):
        share = cm.div(cm.sum(axis=1), axis=0).fillna(0)
        ax.imshow(share.to_numpy(), cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                v = share.iat[i, j]
                ax.text(j, i, f"{v:.2f}" if v >= 0.005 else "·", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.6 else "black")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_title(NAMES.get(name, name), fontsize=10)
    np.atleast_1d(axs)[0].set_ylabel("true (meter-steps, row-normalised)")
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def attack_size_table(data: dict, out: dict, bins=(0, 0.25, 0.5, 1, 2, 4, np.inf)) -> pd.DataFrame:
    """Feeder-level alarm rates on consistent-attack event steps, by size of the faked change."""
    E, R, T, D = data["episodes"], data["readings"], data["true"], data["defs"]
    rows = []
    for ep, e in E[(E.cls == "attack") & (E.subtype == "consistent")].iterrows():
        p = D.index[(D.meter == e.target_meter) & (D.meas_type == "p")][0]
        steps = range(e.event_start, len(R.loc[ep]))
        for s in steps:
            rows.append({"dP_kW": abs(R.at[(ep, s), p] - T.at[(ep, s), p]) * 1e3,
                         "chi2": bool(out["wls"].at[(ep, s), "flag"]),
                         "balance": bool(out["bal_flag"].at[(ep, s)])})
    df = pd.DataFrame(rows)
    df["size"] = pd.cut(df.dP_kW, bins, right=False)
    tab = df.groupby("size", observed=True).agg(steps=("chi2", "size"), chi2=("chi2", "mean"),
                                                 balance=("balance", "mean"))
    return tab


def reference_rates(data: dict, out: dict) -> pd.Series:
    """Same alarms on real load-change event steps (should stay near the false-alarm level)."""
    E = data["episodes"]
    idx = [(ep, s) for ep, e in E[E.cls == "load_change"].iterrows()
           for s in range(e.event_start, 96)]
    return pd.Series({"chi2": out["wls"].flag.loc[idx].mean(),
                      "balance": out["bal_flag"].loc[idx].mean()})


def plot_attack_size(tab: pd.DataFrame, ref: pd.Series, path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tab))
    ax.bar(x - 0.2, tab.chi2, 0.4, label="WLS chi-square alarm", color="tab:blue")
    ax.bar(x + 0.2, tab.balance, 0.4, label="energy-balance alarm", color="tab:orange")
    ax.axhline(ref.chi2, color="tab:blue", ls=":", lw=1, label="chi-square on real load changes")
    ax.axhline(ref.balance, color="tab:orange", ls=":", lw=1, label="balance on real load changes")
    ax.set_xticks(x, [f"{iv.left:g}-{iv.right:g}\n(n={n})" for iv, n in zip(tab.index, tab.steps)], fontsize=8)
    ax.set_xlabel("size of the faked change at the target meter (kW)")
    ax.set_ylabel("share of attack steps flagged"); ax.set_ylim(0, 1)
    ax.set_title("Consistent attack: both tests see it only through the transformer meter")
    ax.legend(fontsize=8); fig.tight_layout()
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def peer_coverage(data: dict, out: dict) -> pd.DataFrame:
    """Share of *healthy* customer meters whose peer check still works after the event."""
    E, L = data["episodes"], data["labels"]
    cov = out["peer_covered"]
    rows = []
    for ep, e in E[E.cls.isin(["normal", "comm_loss"])].iterrows():
        t0 = e.event_start if e.event_start >= 0 else 48
        healthy = [m for m in cov.columns if m not in e.affected.split(",")]
        rows.append({"episode": ep, "case": "normal" if e.cls == "normal" else e.subtype,
                     "gateway": e.gateway, "coverage": cov.loc[ep].loc[t0:, healthy].to_numpy().mean()})
    return pd.DataFrame(rows).set_index("episode")


def plot_peer_coverage(cov: pd.DataFrame, path=None):
    order = ["normal", "dropout", "stale", "gateway"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    data = [cov.coverage[cov.case == c] for c in order]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(order) + 1), ["normal", "one meter\ndrops out", "one meter\ngoes stale",
                                             "whole gateway\nout"])
    ax.set_ylabel("healthy meters with all 5 peers available")
    ax.set_ylim(0, 1.05)
    ax.set_title("Peer check loses its reference when the gateway fails")
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def chance_detection(episodes: pd.DataFrame, pred: pd.DataFrame, start: int = 48, seed: int = 0) -> pd.Series:
    """How often would `detection` score a hit by luck?

    For every normal episode, pick a random customer meter and pretend an event starts at
    `start`; report the share of those where the detector still gives each class at least
    once afterwards. A detection rate is only meaningful well above this level.
    """
    rng = np.random.default_rng(seed)
    meters = [m for m in pred.columns if m != "TRAFO"]
    hits = {c: [] for c in CLASSES if c != "normal"}
    for ep in episodes.index[episodes.cls == "normal"]:
        after = pred.loc[ep].loc[start:, rng.choice(meters)]
        for c in hits:
            hits[c].append((after == c).any())
    return pd.Series({c: np.mean(v) for c, v in hits.items()})


def plot_importance(imp: pd.Series, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    imp.sort_values().plot.barh(ax=ax, color="tab:purple")
    ax.set_xlabel("drop in macro-F1 when the feature is shuffled")
    ax.set_title("What the classifier relies on")
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
