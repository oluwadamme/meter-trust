"""Labelled scenario generator: the five classes, plus a correlated gateway outage.

The dataset is a set of short *episodes*. Each episode is a window of the 4-week base
simulation (default one day, 96 steps) with at most one event in it. The event starts
part-way through the window and lasts to its end, so every episode has a clean lead-in
and detection delay can be measured from a known start.

| class       | subtype         | what changes                                            |
|-------------|-----------------|---------------------------------------------------------|
| normal      | none            | nothing: true values plus meter noise                   |
| load_change | up / down       | one customer's demand is scaled; power flow is re-run,  |
|             |                 | so every reading (transformer included) stays consistent|
| meter_fault | gain / stuck /  | only the faulty meter's reported values: gain 0.5-0.8x, |
|             | drift           | frozen at the last value, or gain drifting to 0.6-0.8x  |
| comm_loss   | dropout / stale | one meter: readings missing, or the last reading        |
|             | / gateway       | repeated; gateway: every meter behind one concentrator  |
|             |                 | goes missing together                                   |
| attack      | naive /         | naive: a random offset on one meter's P. consistent:    |
|             | consistent      | the attacker scales one load in a copy of the grid,     |
|             |                 | runs power flow and writes that copy's values (with     |
|             |                 | noise) into the target meter and its gateway's voltage  |
|             |                 | meter. The transformer meter is never touched.          |

Assumptions to state in the README: the transformer meter is trusted and outside the
attacker's reach; the attacker knows the grid model and can run a power flow; the
head-end knows whether a reading arrived fresh at each step (`fresh`), as a real AMI
head-end does from timestamps. A stuck meter keeps sending fresh readings, while a
stale link does not, which is the only thing that separates the two.

Labels are per meter and per step: "normal" before the event and for meters the
event does not touch. For a real load change only the customer whose demand changed
is labelled; other readings that move with it (voltages, the transformer) are correct
and stay "normal".

Noise is drawn with std from the *true* values (that is the physical meter error).
Estimation should take its weights from the *reported* values, as an operator would:
`measurements.std_devs(defs, readings)`.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import grid
import measurements as meas

CLASSES = ["normal", "load_change", "meter_fault", "comm_loss", "attack"]
SUBTYPES = {
    "normal": ["none"],
    "load_change": ["up", "down"],
    "meter_fault": ["gain", "stuck", "drift"],
    "comm_loss": ["dropout", "stale", "gateway"],
    "attack": ["naive", "consistent"],
}
TRAIN_DAYS = 21  # days 0-20 train, 21-27 test: split by time, never at random


def __getattr__(name):  # OUT_DIR follows the active run (grid.use_run)
    if name == "OUT_DIR":
        return grid.DATA_DIR / "scenarios"
    raise AttributeError(name)


@dataclass
class Episode:
    episode: int
    cls: str
    subtype: str
    split: str
    base_start: int          # first step of the window in the 4-week base run
    event_start: int         # step within the episode where the event begins (-1: none)
    target_meter: str = ""   # main meter hit (customer bus meter "C<bus>")
    load: int = -1           # load index scaled (load_change, consistent attack)
    gateway: str = ""
    factor: float = np.nan   # demand scale / gain / end gain / offset in kW
    affected: list = field(default_factory=list)  # meters labelled with `cls`
    sgen: int = -1           # PV unit that trips (pv_trip)
    cloud_steps: int = 0     # length of a passing cloud (cloud, cloud_attack)
    cloud_factor: float = np.nan  # PV output during the cloud, as a share of normal


def _window(profiles: dict, start: int, stop: int) -> dict:
    return {k: df.iloc[start:stop].reset_index(drop=True) for k, df in profiles.items()}


def _pf_with_changes(net, defs, profiles, start, stop, load=None, factor=1.0,
                     pv_units=(), pv_scale=None) -> pd.DataFrame:
    """True measurement values for base steps start..stop-1 after physical changes.

    `load` is scaled by `factor` (P and Q) for the whole window; each PV unit in
    `pv_units` is multiplied by `pv_scale`, an array with one value per step.
    """
    prof = _window(profiles, start, stop)
    if load is not None:
        for col in ("p_mw", "q_mvar"):
            prof[("load", col)] = prof[("load", col)].copy()
            prof[("load", col)][load] *= factor
    if len(pv_units):
        pv = prof[("sgen", "p_mw")].copy()
        for u in pv_units:
            pv[u] = pv[u].to_numpy() * pv_scale
        prof[("sgen", "p_mw")] = pv
    res = grid.run_power_flow_series(net, prof)
    return meas.true_values(defs, res)


def _pf_with_scaled_load(net, defs, profiles, start, stop, load, factor) -> pd.DataFrame:
    """True measurement values for base steps start..stop-1 with one load scaled."""
    return _pf_with_changes(net, defs, profiles, start, stop, load=load, factor=factor)


def _demand_factor(rng) -> tuple[str, float]:
    if rng.random() < 0.5:
        return "up", float(rng.uniform(1.5, 3.0))     # new appliance, EV, more trade
    return "down", float(rng.uniform(0.2, 0.6))       # empty shop, holiday


def plan_episodes(net, n_per_class: int = 60, n_steps: int = 96, n_days: int = 28,
                  seed: int = 0) -> list[Episode]:
    """Decide every episode's class, timing and target. No simulation yet."""
    rng = np.random.default_rng(seed)
    customer_meters = sorted({f"C{b}" for b in net.load.bus} | {f"C{b}" for b in net.sgen.bus})
    gw_of_meter = {f"C{b}": g for b, g in zip(net.load.bus, net.load.gateway)}
    gw_of_meter.update({f"C{b}": g for b, g in zip(net.sgen.bus, net.sgen.gateway)})
    total = n_days * grid.STEPS_PER_DAY
    episodes = []
    for cls in CLASSES:
        for k in range(n_per_class):
            subtype = SUBTYPES[cls][k % len(SUBTYPES[cls])]
            # every 4th episode *of each subtype* goes to the test weeks
            split = "test" if (k // len(SUBTYPES[cls])) % 4 == 3 else "train"
            lo, hi = (0, TRAIN_DAYS * grid.STEPS_PER_DAY - n_steps) if split == "train" \
                else (TRAIN_DAYS * grid.STEPS_PER_DAY, total - n_steps)
            ep = Episode(len(episodes), cls, subtype, split,
                         base_start=int(rng.integers(lo, hi + 1)),
                         event_start=-1 if cls == "normal" else int(rng.integers(n_steps // 4, 3 * n_steps // 4)))
            if cls == "normal":
                pass
            elif cls == "load_change" or (cls == "attack" and subtype == "consistent"):
                ep.load = int(rng.choice(net.load.index))
                ep.target_meter = f"C{net.load.bus[ep.load]}"
                direction, ep.factor = _demand_factor(rng)
                if cls == "load_change":
                    ep.subtype = direction
            elif cls == "comm_loss" and subtype == "gateway":
                ep.gateway = str(rng.choice(sorted(set(gw_of_meter.values()))))
            else:
                ep.target_meter = str(rng.choice(customer_meters))
                if subtype == "gain":
                    ep.factor = float(rng.uniform(0.5, 0.8))
                elif subtype == "drift":
                    ep.factor = float(rng.uniform(0.6, 0.8))
                elif subtype == "naive":
                    ep.factor = float(rng.choice([-1, 1]) * rng.uniform(0.5, 2.0))  # kW
            if ep.target_meter:
                ep.gateway = ep.gateway or gw_of_meter[ep.target_meter]
            episodes.append(ep)
    return episodes


# --- Rooftop-solar cases (summer_pv run only) ----------------------------------------------
# | subtype      | class       | what happens                                                  |
# | cloud        | normal      | PV on one cable drops to 20-50 % for 30 min-2 h; power flow   |
# |              |             | re-run. A real, harmless event: no meter should be flagged.   |
# | pv_trip      | load_change | one PV inverter trips to zero for the rest of the day. The    |
# |              |             | readings are true, so for the operator this is a real change. |
# | cloud_attack | attack      | a consistent attack (as above) that starts at the same moment |
# |              |             | as a cloud on the target's cable, to hide in the transient.   |
PV_SUBTYPES = {"cloud": "normal", "pv_trip": "load_change", "cloud_attack": "attack"}


def _daytime_start(rng, lo, hi, n_steps, first_hour=9, last_hour=15):
    """A window start and an event step inside it that falls between first_hour and last_hour."""
    while True:
        base = int(rng.integers(lo, hi + 1))
        steps = [t for t in range(n_steps // 4, 3 * n_steps // 4)
                 if first_hour <= ((base + t) % grid.STEPS_PER_DAY) / 4 < last_hour]
        if steps:
            return base, int(rng.choice(steps))


def plan_pv_episodes(net, first_id: int, n_per_subtype: int = 20, n_steps: int = 96,
                     n_days: int = 28, seed: int = 0) -> list[Episode]:
    rng = np.random.default_rng(seed + 2)
    pv_gateways = sorted(set(net.sgen.gateway.dropna()))
    total = n_days * grid.STEPS_PER_DAY
    episodes = []
    for subtype, cls in PV_SUBTYPES.items():
        for k in range(n_per_subtype):
            split = "test" if k % 4 == 3 else "train"
            lo, hi = (0, TRAIN_DAYS * grid.STEPS_PER_DAY - n_steps) if split == "train" \
                else (TRAIN_DAYS * grid.STEPS_PER_DAY, total - n_steps)
            base, t0 = _daytime_start(rng, lo, hi, n_steps)
            ep = Episode(first_id + len(episodes), cls, subtype, split, base_start=base, event_start=t0)
            if subtype == "pv_trip":
                ep.sgen = int(rng.choice(net.sgen.index))
                ep.target_meter = f"C{net.sgen.bus[ep.sgen]}"
                ep.gateway = str(net.sgen.gateway[ep.sgen])
                ep.factor = 0.0
            else:
                ep.gateway = str(rng.choice(pv_gateways))
                ep.cloud_steps = int(rng.integers(2, 9))
                ep.cloud_factor = float(rng.uniform(0.2, 0.5))
                if subtype == "cloud_attack":
                    loads = net.load.index[net.load.gateway == ep.gateway]
                    ep.load = int(rng.choice(loads))
                    ep.target_meter = f"C{net.load.bus[ep.load]}"
                    _, ep.factor = _demand_factor(rng)
            episodes.append(ep)
    return episodes


def _cloud(net, ep, length):
    """PV units behind the episode's gateway, and their output multiplier per step."""
    units = list(net.sgen.index[net.sgen.gateway == ep.gateway])
    scale = np.ones(length)
    scale[: ep.cloud_steps] = ep.cloud_factor
    return units, scale


def simulate_episode(ep: Episode, net, defs, profiles, true_base, n_steps, rng):
    """Return (true, reported, fresh, labels) frames for one episode, steps 0..n_steps-1."""
    a, b, t0 = ep.base_start, ep.base_start + n_steps, ep.event_start
    true = true_base.iloc[a:b].reset_index(drop=True).copy()
    if ep.cls == "load_change" and ep.subtype in ("up", "down"):
        true.iloc[t0:] = _pf_with_scaled_load(net, defs, profiles, a + t0, b,
                                              ep.load, ep.factor).to_numpy()
    elif ep.subtype == "pv_trip":
        true.iloc[t0:] = _pf_with_changes(net, defs, profiles, a + t0, b, pv_units=[ep.sgen],
                                          pv_scale=np.zeros(b - a - t0)).to_numpy()
    elif ep.subtype in ("cloud", "cloud_attack"):
        units, scale = _cloud(net, ep, b - a - t0)
        true.iloc[t0:] = _pf_with_changes(net, defs, profiles, a + t0, b, pv_units=units,
                                          pv_scale=scale).to_numpy()
    reported = meas.add_noise(true, meas.std_devs(defs, true), seed=int(rng.integers(2**31)))
    fresh = pd.DataFrame(True, index=true.index, columns=true.columns)

    meters = ["TRAFO"] + sorted(m for m in defs.meter.unique() if m != "TRAFO")
    labels = pd.DataFrame("normal", index=true.index, columns=meters)
    cols = defs.index[defs.meter == ep.target_meter]          # readings of the target
    pq = defs.index[(defs.meter == ep.target_meter) & defs.meas_type.isin(["p", "q"])]
    after = slice(t0, None)

    if ep.cls == "meter_fault":
        if ep.subtype == "gain":
            reported.loc[t0:, pq] *= ep.factor
        elif ep.subtype == "stuck":
            reported.loc[t0:, cols] = reported.loc[t0 - 1, cols].to_numpy()
        elif ep.subtype == "drift":
            ramp = np.linspace(1.0, ep.factor, n_steps - t0)[:, None]
            reported.loc[t0:, pq] = reported.loc[t0:, pq].to_numpy() * ramp
        ep.affected = [ep.target_meter]
    elif ep.cls == "comm_loss":
        if ep.subtype == "gateway":
            cols = defs.index[defs.gateway == ep.gateway]
            ep.affected = sorted(defs.meter[cols].unique())
        else:
            ep.affected = [ep.target_meter]
        if ep.subtype == "stale":
            reported.loc[t0:, cols] = reported.loc[t0 - 1, cols].to_numpy()
        else:
            reported.loc[t0:, cols] = np.nan
        fresh.loc[t0:, cols] = False
    elif ep.cls == "attack" and ep.subtype == "naive":
        p = defs.index[(defs.meter == ep.target_meter) & (defs.meas_type == "p")]
        reported.loc[t0:, p] += ep.factor / 1e3  # kW -> MW
        ep.affected = [ep.target_meter]
    elif ep.cls == "attack" and ep.subtype in ("consistent", "cloud_attack"):
        if ep.subtype == "cloud_attack":  # the fake operating point includes the real cloud
            units, scale = _cloud(net, ep, b - a - t0)
            fake = _pf_with_changes(net, defs, profiles, a + t0, b, load=ep.load, factor=ep.factor,
                                    pv_units=units, pv_scale=scale)
        else:
            fake = _pf_with_scaled_load(net, defs, profiles, a + t0, b, ep.load, ep.factor)
        fake_noisy = meas.add_noise(fake, meas.std_devs(defs, fake), seed=int(rng.integers(2**31)))
        v_gw = defs.index[(defs.meas_type == "v") & (defs.gateway == ep.gateway)]
        hit = cols.union(v_gw)
        reported.loc[t0:, hit] = fake_noisy[hit].to_numpy()
        ep.affected = sorted(defs.meter[hit].unique())
    elif ep.cls == "load_change":
        ep.affected = [ep.target_meter]

    if ep.affected:
        labels.loc[after, ep.affected] = ep.cls
    return true, reported, fresh, labels


def generate(net=None, n_per_class: int = 60, n_steps: int = 96, seed: int = 0,
             pv_extras: bool | None = None) -> dict:
    """Build the whole dataset in memory. ~1-2 min for the default 300 episodes.

    pv_extras adds 20 episodes each of cloud, pv_trip and cloud_attack after the 300
    standard ones (default: only for the summer_pv run). The standard episodes come out
    identical either way.
    """
    pv_extras = (grid.RUN == "summer_pv") if pv_extras is None else pv_extras
    net = net if net is not None else grid.load_net()
    defs = meas.define_measurements(net)
    profiles = grid.get_profiles(net, n_days=28)
    pf_dir = grid.DATA_DIR / "pf"
    base = grid.load_results(pf_dir) if pf_dir.exists() else None
    if not base:
        base = grid.run_power_flow_series(net, profiles)
        grid.save_results(base, pf_dir)
    true_base = meas.true_values(defs, base)

    episodes = plan_episodes(net, n_per_class, n_steps, seed=seed)
    if pv_extras:
        episodes += plan_pv_episodes(net, len(episodes), n_steps=n_steps, seed=seed)
    rng = np.random.default_rng(seed + 1)
    parts = {"true": [], "readings": [], "fresh": [], "labels": []}
    for ep in episodes:
        frames = simulate_episode(ep, net, defs, profiles, true_base, n_steps, rng)
        for name, df in zip(parts, frames):
            df = df.copy()
            df.index = pd.MultiIndex.from_product([[ep.episode], df.index], names=["episode", "step"])
            parts[name].append(df)
    data = {name: pd.concat(dfs) for name, dfs in parts.items()}
    eps = pd.DataFrame([vars(e) for e in episodes]).set_index("episode")
    eps["affected"] = eps.affected.map(lambda m: ",".join(m))
    data["episodes"] = eps
    data["defs"] = defs
    return data


def save(data: dict, folder: Path | None = None) -> None:
    folder = folder or grid.DATA_DIR / "scenarios"
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.rename(columns=str).to_parquet(folder / f"{name}.parquet")


def load(folder: Path | None = None) -> dict:
    folder = folder or grid.DATA_DIR / "scenarios"
    data = {}
    for f in sorted(folder.glob("*.parquet")):
        df = pd.read_parquet(f)
        if f.stem in ("true", "readings", "fresh"):
            df.columns = df.columns.astype(int)  # measurement ids
        data[f.stem] = df
    return data


if __name__ == "__main__":
    import time
    t = time.time()
    d = generate()
    save(d)
    print(f"{len(d['episodes'])} episodes, {len(d['readings'])} rows in {time.time() - t:.0f}s")
    print(d["episodes"].groupby(["cls", "subtype", "split"]).size().unstack(fill_value=0))
