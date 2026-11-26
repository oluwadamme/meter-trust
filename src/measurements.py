"""Where the meters are, how noisy they are, and the noisy readings over time.

Measurement set (see the project plan):
- transformer LV side: P, Q and bus voltage (the trusted feeder-head meter)
- every customer: P and Q as a bus injection, std 1.5 % of reading
- the last bus of each feeder (one per gateway): voltage magnitude

Customer meters enter the estimator as *bus* P/Q measurements, because pandapower's
estimator only reads bus, line and trafo measurements (load/sgen measurements are
stored but ignored). Where two customers share a bus, the bus meter is their sum.
Signs follow pandapower's res_bus convention: positive P = consumption.
"""
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top

# Standard deviations, relative to the reading, with an absolute floor so that a
# near-zero reading does not get an almost-infinite weight in WLS.
STD = {
    "customer_pq": (0.015, 50e-6),   # 1.5 %, floor 50 W / 50 var
    "trafo_pq": (0.005, 100e-6),     # 0.5 %, floor 100 W
    "v": (0.005, 0.0),               # 0.5 % of 1 pu
}


def define_measurements(net: pp.pandapowerNet) -> pd.DataFrame:
    """Return one row per measurement: what it measures and where the true value lives.

    `source` and `col` point to a column of the power-flow results dict from
    grid.run_power_flow_series (e.g. "res_bus.p_mw", column = bus index).
    """
    rows = []
    trafo = net.trafo.index[0]
    lv_bus = net.trafo.lv_bus.iat[0]
    rows += [
        dict(meas_type="p", element_type="trafo", element=trafo, side="lv",
             source="res_trafo.p_lv_mw", col=trafo, group="trafo_pq", gateway="TRAFO"),
        dict(meas_type="q", element_type="trafo", element=trafo, side="lv",
             source="res_trafo.q_lv_mvar", col=trafo, group="trafo_pq", gateway="TRAFO"),
        dict(meas_type="v", element_type="bus", element=lv_bus, side=None,
             source="res_bus.vm_pu", col=lv_bus, group="v", gateway="TRAFO"),
    ]

    bus_gw = pd.concat([net.load[["bus", "gateway"]], net.sgen[["bus", "gateway"]]])
    bus_gw = bus_gw.drop_duplicates("bus").set_index("bus").gateway.sort_index()
    for bus, gw in bus_gw.items():
        rows.append(dict(meas_type="p", element_type="bus", element=bus, side=None,
                         source="res_bus.p_mw", col=bus, group="customer_pq", gateway=gw))
        rows.append(dict(meas_type="q", element_type="bus", element=bus, side=None,
                         source="res_bus.q_mvar", col=bus, group="customer_pq", gateway=gw))

    # Voltage at the electrically farthest customer bus of each gateway.
    dist = top.calc_distance_to_bus(net, lv_bus)
    for gw, buses in bus_gw.groupby(bus_gw).groups.items():
        far = dist.loc[list(buses)].idxmax()
        rows.append(dict(meas_type="v", element_type="bus", element=far, side=None,
                         source="res_bus.vm_pu", col=far, group="v", gateway=gw))

    defs = pd.DataFrame(rows)
    defs.index.name = "meas_id"
    return defs


def true_values(defs: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Noise-free readings: steps x measurements."""
    cols = {i: results[r.source][r.col].to_numpy() for i, r in defs.iterrows()}
    return pd.DataFrame(cols)


def std_devs(defs: pd.DataFrame, true: pd.DataFrame) -> pd.DataFrame:
    """Per-reading standard deviation (same shape as `true`)."""
    rel = defs.group.map(lambda g: STD[g][0]).to_numpy()
    floor = defs.group.map(lambda g: STD[g][1]).to_numpy()
    return pd.DataFrame(np.maximum(rel * true.abs().to_numpy(), floor),
                        index=true.index, columns=true.columns)


def add_noise(true: pd.DataFrame, std: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return true + rng.standard_normal(true.shape) * std


def write_to_net(net: pp.pandapowerNet, defs: pd.DataFrame, values: pd.Series,
                 std: pd.Series) -> None:
    """Replace net.measurement with the readings of one time step.

    NaN readings (lost or missing data) are left out, as an operator would do.
    """
    net.measurement.drop(net.measurement.index, inplace=True)
    for i, r in defs.iterrows():
        if np.isnan(values[i]):
            continue
        pp.create_measurement(net, r.meas_type, r.element_type, float(values[i]),
                              float(std[i]), int(r.element), side=r.side, index=int(i))
