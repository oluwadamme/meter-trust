"""Load the SimBench LV feeder, its 15-minute profiles, and group customers into gateways.

Two runs share all the code (pick one with `use_run` or the METER_TRUST_RUN variable):

- "winter"     SimBench `1-LV-semiurb4--0-sw`, the first 4 weeks of January. One 0.4 MVA
               transformer, 44 buses, 41 loads, one small PV unit. Data in data/.
- "summer_pv"  `1-LV-semiurb4--2-sw`, SimBench's 2034 scenario of the same feeder, 4
               weeks from 1 June: 58 loads and 6 rooftop PV units (172 kWp against
               337 kW of load), so power flows back through the transformer at midday.
               Its 4 home batteries are switched off: they would smooth exactly the
               signals being tested, and their control is a separate question.
               Data in data/summer_pv/.

Public test grid, not utility data.
"""
import copy
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top
import simbench as sb
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries

STEPS_PER_DAY = 96  # 15-minute resolution
ROOT_DATA = Path(__file__).resolve().parents[1] / "data"
RUNS = {
    "winter": dict(code="1-LV-semiurb4--0-sw", start_day=0, pv_variation=0.0,
                   data_dir=ROOT_DATA),
    "summer_pv": dict(code="1-LV-semiurb4--2-sw", start_day=152, pv_variation=0.1,
                      data_dir=ROOT_DATA / "summer_pv"),  # day 152 = 1 June 2016
}


def use_run(name: str) -> None:
    """Switch every module to one run's grid, season and data folder."""
    global RUN, GRID_CODE, START_DAY, PV_VARIATION, DATA_DIR
    cfg = RUNS[name]
    RUN, GRID_CODE, START_DAY = name, cfg["code"], cfg["start_day"]
    PV_VARIATION, DATA_DIR = cfg["pv_variation"], cfg["data_dir"]


use_run(os.environ.get("METER_TRUST_RUN", "winter"))


def load_net(code: str | None = None) -> pp.pandapowerNet:
    """Return the SimBench net with a `gateway` column added to net.load and net.sgen."""
    net = sb.get_simbench_net(code or GRID_CODE)
    if len(net.storage):
        net.storage["in_service"] = False
    assign_gateways(net)
    return net


def assign_gateways(net: pp.pandapowerNet) -> pd.Series:
    """One gateway (data concentrator) per LV feeder leaving the transformer busbar.

    A real concentrator usually serves one feeder, so customers behind the same gateway
    also share a physical path. Feeders with no customers get no gateway. Returns the
    bus -> gateway map and writes `gateway` onto net.load and net.sgen.
    """
    lv_bus = net.trafo.lv_bus.iat[0]
    g = top.create_nxgraph(net, include_trafos=False)
    g.remove_node(lv_bus)
    customer_buses = set(net.load.bus) | set(net.sgen.bus)
    feeders = [c for c in nx.connected_components(g) if c & customer_buses]
    feeders.sort(key=lambda c: min(c))  # stable order
    bus_to_gw = {b: f"GW{i + 1}" for i, comp in enumerate(feeders) for b in comp}
    net.load["gateway"] = net.load.bus.map(bus_to_gw)
    net.sgen["gateway"] = net.sgen.bus.map(bus_to_gw)
    return pd.Series(bus_to_gw, name="gateway")


def get_profiles(net: pp.pandapowerNet, n_days: int = 28, start_day: int | None = None,
                 variation: float = 0.3, seed: int = 0, pv_variation: float | None = None) -> dict:
    """Absolute load/PV profiles (MW, Mvar) for `n_days` days, re-indexed from 0.

    Keys are (element, column), e.g. ("load", "p_mw"), as returned by SimBench.

    SimBench gives this feeder only 10 distinct load profiles for 41 customers, so most
    customers are exact scaled copies of another one. Real neighbours are never that
    alike, and a peer check would look far better than it is. Each customer's P and Q
    are therefore multiplied by its own smooth random factor (log-normal AR(1), std
    `variation`, about 1 hour memory). Set variation=0 for the raw SimBench profiles.
    PV units likewise get `pv_variation` (roof direction, shading); SimBench gives all
    PV units on this feeder the same profile. Defaults come from the active run.
    """
    start_day = START_DAY if start_day is None else start_day
    pv_variation = PV_VARIATION if pv_variation is None else pv_variation
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    start, stop = start_day * STEPS_PER_DAY, (start_day + n_days) * STEPS_PER_DAY
    out = {}
    for key, df in profiles.items():
        part = df.iloc[start:stop].reset_index(drop=True)
        out[key] = part
    if variation > 0:
        factor = _customer_variation(len(out[("load", "p_mw")]), out[("load", "p_mw")].columns,
                                     variation, seed)
        for col in ("p_mw", "q_mvar"):
            out[("load", col)] = out[("load", col)] * factor
    if pv_variation > 0 and ("sgen", "p_mw") in out:
        pv = out[("sgen", "p_mw")]
        out[("sgen", "p_mw")] = pv * _customer_variation(len(pv), pv.columns, pv_variation, seed + 1)
    return out


def _customer_variation(n_steps: int, loads, sigma: float, seed: int) -> pd.DataFrame:
    """Mean-one log-normal AR(1) factors, one column per load."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-1 / 4)  # correlation time of 4 steps = 1 hour
    x = np.zeros((n_steps, len(loads)))
    x[0] = rng.standard_normal(len(loads))
    eps = rng.standard_normal((n_steps, len(loads))) * np.sqrt(1 - phi ** 2)
    for t in range(1, n_steps):
        x[t] = phi * x[t - 1] + eps[t]
    return pd.DataFrame(np.exp(sigma * x - sigma ** 2 / 2), columns=loads)


def run_power_flow_series(net: pp.pandapowerNet, profiles: dict) -> dict:
    """Time-series AC power flow driven by the profiles. Returns result DataFrames.

    Only load P/Q and sgen P are driven; the MV grid behind the transformer is the
    fixed ext_grid voltage from SimBench.
    """
    net = copy.deepcopy(net)
    n_steps = len(profiles[("load", "p_mw")])
    for (elm, col), df in profiles.items():
        if elm not in ("load", "sgen") or df.shape[1] == 0:
            continue
        if elm == "sgen" and col != "p_mw":
            continue
        ConstControl(net, element=elm, variable=col, element_index=df.columns,
                     data_source=DFData(df), profile_name=df.columns)

    ow = OutputWriter(net, time_steps=range(n_steps), output_path=None)
    for table, var in [("res_bus", "vm_pu"), ("res_bus", "va_degree"),
                       ("res_bus", "p_mw"), ("res_bus", "q_mvar"),
                       ("res_load", "p_mw"), ("res_load", "q_mvar"), ("res_sgen", "p_mw"),
                       ("res_trafo", "p_lv_mw"), ("res_trafo", "q_lv_mvar"),
                       ("res_trafo", "p_hv_mw"), ("res_trafo", "q_hv_mvar"),
                       ("res_line", "pl_mw")]:
        ow.log_variable(table, var)
    run_timeseries(net, time_steps=range(n_steps), verbose=False)
    return {f"{k}": v.copy() for k, v in ow.output.items() if k != "Parameters"}


def save_results(results: dict, folder: Path | None = None) -> None:
    folder = folder or DATA_DIR / "pf"
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        # parquet needs string column names; rename a copy, not the caller's frame
        df.rename(columns=str).to_parquet(folder / f"{name.replace('.', '__')}.parquet")


def load_results(folder: Path | None = None) -> dict:
    folder = folder or DATA_DIR / "pf"
    out = {}
    for f in sorted(folder.glob("*.parquet")):
        df = pd.read_parquet(f)
        df.columns = df.columns.astype(int)
        out[f.stem.replace("__", ".")] = df
    return out


if __name__ == "__main__":
    import time
    net = load_net()
    prof = get_profiles(net, n_days=28)
    t0 = time.time()
    res = run_power_flow_series(net, prof)
    save_results(res)
    vm = res["res_bus.vm_pu"]
    print(f"{len(vm)} steps in {time.time() - t0:.0f}s; "
          f"V range {vm.min().min():.3f}-{vm.max().max():.3f} pu; "
          f"peak trafo P {res['res_trafo.p_lv_mw'].abs().max().max() * 1e3:.0f} kW")
    print(net.load.gateway.value_counts().sort_index().to_dict())
