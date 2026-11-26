"""WLS state estimation (pandapower) and a chi-square bad-data test on its residuals.

API checked against pandapower 3.5.6:
    estimate(net, algorithm="wls", init="flat", ..., zero_injection=...) -> bool
        writes net.res_bus_est, net.res_trafo_est, net.res_line_est
    chi2_analysis(net, ..., chi2_prob_false=0.05) -> bool
    remove_bad_data(net, ..., rn_max_threshold=3.0) -> bool

Two pandapower 3.5.6 problems shape this file:
1. zero_injection="no_inj_bus" raises an IndexError on SimBench nets (it combines two
   boolean Series with different indexes), and the default "aux_bus" adds no
   constraints here, which leaves the feeder unobservable. So the zero-injection
   buses are passed as an explicit list.
2. chi2_analysis and remove_bad_data take no zero_injection argument, and internally
   call estimate(v, delta, calculate_voltage_angles) positionally, so the angle flag
   lands in the zero_injection slot. On this feeder they fail as unobservable.
   The chi-square test is therefore computed here from the WLS residuals. The
   largest-normalised-residual test needs the residual covariance and is left for
   Week 3 (detectors).

The estimator uses algorithm="wls_with_zero_constraint", which enforces P = Q = 0 at
junction buses exactly. Plain "wls" models them as pseudo-measurements with a fixed
1 kW std, looser than the customer meters, so they soak up residual and J runs low
(mean 3.1 instead of 5 on clean data). With exact constraints the mean J matches
its degrees of freedom.
"""
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.estimation import estimate
from scipy.stats import chi2

from measurements import write_to_net

# Where the estimated value of each measurement type lives after estimate().
_EST_LOOKUP = {
    ("bus", "v", None): ("res_bus_est", "vm_pu"),
    ("bus", "p", None): ("res_bus_est", "p_mw"),
    ("bus", "q", None): ("res_bus_est", "q_mvar"),
    ("trafo", "p", "lv"): ("res_trafo_est", "p_lv_mw"),
    ("trafo", "q", "lv"): ("res_trafo_est", "q_lv_mvar"),
    ("trafo", "p", "hv"): ("res_trafo_est", "p_hv_mw"),
    ("trafo", "q", "hv"): ("res_trafo_est", "q_hv_mvar"),
}


def zero_injection_buses(net: pp.pandapowerNet) -> list:
    """Buses with no load, PV or grid connection: P = Q = 0 is known exactly."""
    busy = set(net.load.bus) | set(net.sgen.bus) | set(net.ext_grid.bus)
    return [b for b in net.bus.index if b not in busy]


def run_wls(net: pp.pandapowerNet, defs: pd.DataFrame, values: pd.Series,
            std: pd.Series) -> pd.DataFrame | None:
    """Estimate the state for one time step. Returns res_bus_est (or None if it fails)."""
    write_to_net(net, defs, values, std)
    ok = estimate(net, algorithm="wls_with_zero_constraint", init="flat",
                  zero_injection=zero_injection_buses(net))
    return net.res_bus_est.copy() if ok else None


def residuals(net: pp.pandapowerNet) -> pd.DataFrame:
    """Measured minus estimated value for every row of net.measurement (after run_wls)."""
    m = net.measurement
    est = []
    for _, r in m.iterrows():
        side = r.side if r.element_type != "bus" else None
        table, col = _EST_LOOKUP[(r.element_type, r.measurement_type, side)]
        est.append(net[table].at[r.element, col])
    out = pd.DataFrame({"z": m.value, "h": est, "std": m.std_dev}, index=m.index)
    out["r_w"] = (out.z - out.h) / out["std"]  # weighted residual
    return out


def degrees_of_freedom(net: pp.pandapowerNet) -> int:
    """Measurements plus 2 zero-injection constraints per junction bus, minus states (2N - 1).

    About 5 here: customer meters fix every injection, so the only redundancy is the
    transformer P/Q and the few voltage readings. That low redundancy matters for
    which errors the residual test can see.
    """
    m = len(net.measurement) + 2 * len(zero_injection_buses(net))
    return m - (2 * len(net.bus) - 1)


def chi2_test(net: pp.pandapowerNet, prob_false: float = 0.05) -> dict:
    """Chi-square test on J = sum(r_w^2). Call after run_wls. flag=True: bad data suspected."""
    j = float((residuals(net).r_w ** 2).sum())
    dof = degrees_of_freedom(net)
    threshold = float(chi2.ppf(1 - prob_false, dof))
    return {"J": j, "dof": dof, "threshold": threshold, "flag": j > threshold}
