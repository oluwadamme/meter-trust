"""Load the SimBench LV feeder, its 15-minute profiles, and group customers into gateways.

The feeder is SimBench `1-LV-semiurb4--0-sw`: one 0.4 MVA MV/LV transformer, 44 buses,
41 loads (households and small businesses) and one small PV unit. It is a public test
grid, not utility data.
"""
import copy
from pathlib import Path

import networkx as nx
import pandas as pd
import pandapower as pp
import pandapower.topology as top
import simbench as sb
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries

GRID_CODE = "1-LV-semiurb4--0-sw"
STEPS_PER_DAY = 96  # 15-minute resolution
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_net(code: str = GRID_CODE) -> pp.pandapowerNet:
    """Return the SimBench net with a `gateway` column added to net.load and net.sgen."""
    net = sb.get_simbench_net(code)
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


def get_profiles(net: pp.pandapowerNet, n_days: int = 28, start_day: int = 0) -> dict:
    """Absolute load/PV profiles (MW, Mvar) for `n_days` days, re-indexed from 0.

    Keys are (element, column), e.g. ("load", "p_mw"), as returned by SimBench.
    """
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    start, stop = start_day * STEPS_PER_DAY, (start_day + n_days) * STEPS_PER_DAY
    out = {}
    for key, df in profiles.items():
        part = df.iloc[start:stop].reset_index(drop=True)
        out[key] = part
    return out


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


def save_results(results: dict, folder: Path = DATA_DIR / "pf") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        # parquet needs string column names; rename a copy, not the caller's frame
        df.rename(columns=str).to_parquet(folder / f"{name.replace('.', '__')}.parquet")


def load_results(folder: Path = DATA_DIR / "pf") -> dict:
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
