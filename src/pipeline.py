"""Run the whole chain for one run ("winter" or "summer_pv"), rebuilding missing data.

    data, out = pipeline.run("summer_pv")

Power flow and scenarios take about a minute; the WLS baseline about 6 minutes on 8
cores the first time (then it is read from data/.../detectors/wls.parquet).
"""
import numpy as np
import pandas as pd

import detectors as det
import evaluate as ev
import grid
import scenarios as sc
import train as tr


def run(name: str) -> tuple[dict, dict]:
    grid.use_run(name)
    if not (grid.DATA_DIR / "scenarios" / "labels.parquet").exists():
        sc.save(sc.generate())          # also builds the power flow if missing
    data = sc.load()
    wls_file = grid.DATA_DIR / "detectors" / "wls.parquet"
    if wls_file.exists():
        wls = pd.read_parquet(wls_file)
    else:
        wls = det.run_wls_baseline(data)
        wls_file.parent.mkdir(parents=True, exist_ok=True)
        wls.to_parquet(wls_file)
    out = det.run_all(data, wls=wls)
    feat = tr.build_features(data, out)
    model, threshold = tr.train(feat, data["episodes"])
    out["pred"]["ml"] = tr.predict_frame(model, threshold, feat, data["labels"])
    out["model"], out["threshold"], out["feat"] = model, threshold, feat
    return data, out


def summary(data: dict, out: dict) -> pd.Series:
    """Headline numbers for one run, for side-by-side comparison."""
    E, L, wls = data["episodes"], data["labels"], out["wls"]
    normal = wls.loc[E.index[E.cls == "normal"]]
    test = E.index[E.split == "test"]
    s = {
        "WLS: mean J on normal episodes (dof 5)": normal.J.mean(),
        "WLS: chi-square alarm rate on normal": normal.flag.mean(),
        "Energy balance: residual std (kW)": out["bal_model"]["sigma"] * 1e3,
    }
    for name, key in [("Peer + balance", "balance"), ("Classifier", "ml")]:
        d = ev.detection(E.loc[test] if key == "ml" else E, out["pred"][key])
        for cls in ["load_change", "meter_fault", "attack"]:
            s[f"{name}: {cls} events detected"] = d[d.cls == cls].detected.mean()
        s[f"{name}: consistent attacks detected"] = d[d.subtype == "consistent"].detected.mean()
        s[f"{name}: false alarms"] = ev.false_alarm_rate(L.loc[test] if key == "ml" else L,
                                                         out["pred"][key].loc[test] if key == "ml" else out["pred"][key])
    return pd.Series(s)


def peer_quality(data: dict, out: dict) -> pd.DataFrame:
    """Peer-model residual std per meter, PV customers vs the rest."""
    net = grid.load_net()
    pv_meters = {f"C{b}" for b in net.sgen.bus}
    hist = det.history(None, data["defs"])
    pcol = det.meter_columns(data["defs"])
    mean_p = hist[pcol.to_numpy()].abs().mean().set_axis(pcol.index) * 1e3
    rows = [{"meter": m, "PV customer": m in pv_meters, "residual std (kW)": mod["sigma"] * 1e3,
             "residual / mean |P|": mod["sigma"] * 1e3 / mean_p[m]}
            for m, mod in out["peer_model"].items()]
    return pd.DataFrame(rows).groupby("PV customer")[["residual std (kW)", "residual / mean |P|"]].median()
