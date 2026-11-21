"""How much of each unit's groundwater trend is rainfall, and how much is people (step 05, first pass).

    python src/04_climate_vs_pumping.py

Idea. Storage remembers past rainfall: a wet monsoon recharges aquifers for months to years. Following Humphrey et al.
(2017), the climate-driven part of storage is modelled as rainfall anomalies passed through an exponential memory,
    C(t) = sum_j P'(t - j) exp(-j / tau),     P' = monthly precipitation minus its 2002-2024 monthly climatology,
and each member's groundwater series is fitted as
    GWSA(t) = a + s t + d [t > 2018] + annual and semi-annual harmonics + b C(t),
with tau (3 to 60 months) chosen by fit. The climate component b C(t) and the rest (GWSA - b C) are then given their
own trends per period. If the post-2018 slowdown is rain, the climate component should rise after 2018 and the rest
should keep falling at the earlier rate; if it is people (less pumping, more recharge works), the rest itself slows.
Limits: one rainfall product (ERA5-Land, which under-represents irrigation feedbacks); a linear storage response; the
"rest" also contains canal recharge changes and any model error, not only pumping.
Writes outputs/climate_vs_rest.csv and figures/climate_vs_rest.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
P, OUT, FIG = R / "data/processed", R / "outputs", R / "figures"
PERIODS = {"2002-2009": ("2002-04", "2009-12"), "2010-2017": ("2010-01", "2017-06"), "2018-latest": ("2018-06", "2030-01")}
GAP = pd.Timestamp("2018-01-01")


def unit_precip(ur, names):
    """Monthly precipitation anomaly (mm) per unit, area-weighted, as a DataFrame (month x unit)."""
    p = xr.open_dataarray(P / "precip_era5land_0p5.nc")
    lat = p.lat.values; w = np.cos(np.radians(lat))[:, None] * np.ones(ur.shape)
    out = {}
    for u, n in enumerate(names, 1):
        m = ur == u
        out[n] = (p.values[:, m] * w[m]).sum(1) / w[m].sum()
    df = pd.DataFrame(out, index=p.time.to_index())
    base = df[(df.index >= "2002-01") & (df.index < "2025-01")]
    clim = base.groupby(base.index.month).mean()
    return df - clim.loc[df.index.month].values


def memory(pa, tau):
    """Exponentially weighted running sum of precipitation anomalies (memory tau months)."""
    k = np.exp(-np.arange(0, int(5 * tau) + 1) / tau)
    x = np.convolve(np.nan_to_num(pa), k)[: len(pa)]
    return x


def fit_unit(t, y, c):
    """Least-squares fit of the model; returns coefficients, design pieces and R2."""
    ok = np.isfinite(y)
    w = 2 * np.pi * t
    X = np.column_stack([np.ones_like(t), t, (t > (GAP - pd.Timestamp("2002-01-01")).days / 365.25).astype(float),
                         np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w), c])
    b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    r2 = 1 - np.var(y[ok] - X[ok] @ b) / np.var(y[ok])
    return b, X, r2


def trend(t, y):
    """OLS slope (per year) with harmonics, NaN if under 36 months."""
    ok = np.isfinite(y)
    if ok.sum() < 36:
        return np.nan
    w = 2 * np.pi * t[ok]
    X = np.column_stack([np.ones(ok.sum()), t[ok], np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
    return np.linalg.lstsq(X, y[ok], rcond=None)[0][1]


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m3", R / "src/03_ensemble_units.py"); m3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3)
    ur = m3.unit_raster(); names = list(m3.UNITS)
    pa = unit_precip(ur, names)
    mo = pd.read_csv(OUT / "units_monthly.csv"); mo = mo[mo.stores != "CLSM"]; mo["t"] = pd.to_datetime(mo.month)
    rows, show = [], {}
    for (u, gk, sk), d in mo.groupby(["unit", "grace", "stores"]):
        d = d.set_index("t").gwsa_cm
        pidx = pa.index
        best = None
        for tau in (3, 6, 9, 12, 18, 24, 36, 48, 60):
            c = pd.Series(memory(pa[u].values, tau), index=pidx).reindex(d.index).values
            t = (d.index - pd.Timestamp("2002-01-01")).days.values / 365.25
            b, X, r2 = fit_unit(t, d.values, c)
            if best is None or r2 > best[0]:
                best = (r2, tau, b, c, t)
        r2, tau, b, c, t = best
        clim = b[-1] * c; rest = d.values - clim
        rec = {"unit": u, "grace": gk, "stores": sk, "tau_months": tau, "r2": r2, "b": b[-1]}
        for pn, (a0, a1) in PERIODS.items():
            sel = (d.index >= a0) & (d.index <= a1)
            rec[f"total_{pn}"] = trend(t[sel], d.values[sel])
            rec[f"climate_{pn}"] = trend(t[sel], clim[sel])
            rec[f"rest_{pn}"] = trend(t[sel], rest[sel])
        rows.append(rec)
        if gk == "JPL-CRI" and sk == "ERA5-Land":
            show[u] = (d.index, d.values, clim, rest)
    df = pd.DataFrame(rows); df.to_csv(OUT / "climate_vs_rest.csv", index=False)
    cols = [c for c in df.columns if c.split("_")[0] in ("total", "climate", "rest")]
    s = df.groupby("unit")[cols + ["tau_months", "r2"]].median().round(2)
    key = ["Indian Punjab, Haryana, Delhi", "Rajasthan", "Uttar Pradesh, Uttarakhand", "Pakistan Punjab", "Bihar, Jharkhand",
           "Bengal basin (West Bengal, Bangladesh)", "Maharashtra", "Gujarat", "South India (hard rock)"]
    for part in ("total", "climate", "rest"):
        print(f"\n{part} trend, cm/yr (median of members)")
        print(s.loc[key, [f"{part}_{p}" for p in PERIODS]].to_string())
    print("\nmemory (months) and R2:\n", s.loc[key, ["tau_months", "r2"]].to_string())
    figure(show, key[:6])


def figure(show, units):
    """Storage, its rainfall-driven part and the rest, for six units (one ensemble member)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(units), 1, figsize=(10, 2.2 * len(units)), sharex=True)
    for ax, u in zip(axs, units):
        i, y, c, r = show[u]
        sm = lambda v: pd.Series(v, index=i).rolling(12, center=True, min_periods=6).mean()
        ax.plot(i, sm(y), "k", lw=1.5, label="groundwater storage")
        ax.plot(i, sm(c), "tab:blue", lw=1.2, label="rainfall-driven part")
        ax.plot(i, sm(r), "tab:red", lw=1.2, label="the rest (mostly pumping)")
        ax.axvspan(pd.Timestamp("2017-07-01"), pd.Timestamp("2018-05-31"), color="0.88")
        ax.set_title(u, fontsize=9, loc="left"); ax.set_ylabel("cm")
    axs[0].legend(fontsize=7, ncol=3, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "climate_vs_rest.png", dpi=150); plt.close(fig)
    print("figure written")


if __name__ == "__main__":
    main()
