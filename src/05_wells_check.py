"""Check GRACE groundwater trends against CGWB observation wells, unit by unit (India only), 2013-2020.

    python src/05_wells_check.py

Wells: CGWB depth to water level (m below ground), about 23,000 stations, 2013-01 to 2023-03, from the India Data
Portal mirror of India-WRIS (Open Data Commons Attribution License; data/raw/cgwb/cgwb.csv). Measured four times a
year (January, May, August, November); dense to 2020.
Per well: slope of May (pre-monsoon) depth over 2013-2020 and of November (post-monsoon) depth, needing at least 5
years. Per unit: median and interquartile range of well slopes. Storage change = -slope x specific yield Sy, with Sy
ranges typical of the aquifer (alluvium 0.08-0.16; hard rock 0.01-0.03; CGWB GEC-2015 norms).
GRACE: ensemble members of step 03, trend over the same months (2013-01..2020-12) with the seasonal cycle, without a
step at the mission gap (the gap falls inside the window; the result is checked with 2013-2017 alone).
Writes outputs/wells_vs_grace.csv and figures/wells_vs_grace.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]
OUT, FIG = R / "outputs", R / "figures"
HARD = {"South India (hard rock)", "Maharashtra", "Madhya Pradesh, Chhattisgarh", "Odisha", "Bihar, Jharkhand"}   # mostly hard rock or mixed
SY = {"alluvium": (0.08, 0.16), "hard": (0.01, 0.03)}


def well_slopes(d, month):
    """Per-station slope (m/yr, positive = deeper) of depth in one survey month, 2013-2020, >= 5 years."""
    x = d[(d.date.dt.month == month) & (d.date.dt.year.between(2013, 2020))]
    x = x.groupby(["station_name", "latitude", "longitude", x.date.dt.year]).currentlevel.mean().reset_index()
    out = []
    for (s, la, lo), g in x.groupby(["station_name", "latitude", "longitude"]):
        if g.date.nunique() >= 5 and g.currentlevel.notna().sum() >= 5:
            b = np.polyfit(g.date.values.astype(float), g.currentlevel.values, 1)[0]
            out.append({"station": s, "lat": la, "lon": lo, "slope": b})
    return pd.DataFrame(out)


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m3", R / "src/03_ensemble_units.py"); m3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3)
    ur = m3.unit_raster(); names = list(m3.UNITS)
    d = pd.read_csv(R / "data/raw/cgwb/cgwb.csv", low_memory=False, parse_dates=["date"])
    d = d[(d.currentlevel > 0) & (d.currentlevel < 200)]
    rows = []
    mo = pd.read_csv(OUT / "units_monthly.csv"); mo = mo[mo.stores != "CLSM"]; mo["t"] = pd.to_datetime(mo.month)
    for month, label in ((5, "pre-monsoon"), (11, "post-monsoon")):
        w = well_slopes(d, month)
        i = ((w.lat - 5) / 0.5).astype(int).clip(0, ur.shape[0] - 1); j = ((w.lon - 60) / 0.5).astype(int).clip(0, ur.shape[1] - 1)
        w["unit"] = [names[u - 1] if u > 0 else None for u in ur[i.values, j.values]]
        for u, g in w.dropna(subset=["unit"]).groupby("unit"):
            if len(g) < 30:
                continue
            sy = SY["hard" if u in HARD else "alluvium"]
            med = g.slope.median()
            rows.append({"unit": u, "season": label, "wells": len(g), "depth_slope_med_m_yr": med,
                         "depth_slope_q25": g.slope.quantile(.25), "depth_slope_q75": g.slope.quantile(.75),
                         "share_falling": (g.slope > 0).mean(), "aquifer": "hard rock / mixed" if u in HARD else "alluvium",
                         "storage_lo_cm_yr": -med * sy[1] * 100, "storage_hi_cm_yr": -med * sy[0] * 100})
    W = pd.DataFrame(rows)
    gr = []
    for u in W.unit.unique():
        for (gk, sk), g in mo[mo.unit == u].groupby(["grace", "stores"]):
            for lab, (a, b) in (("2013-2020", ("2013-01", "2020-12")), ("2013-2017", ("2013-01", "2017-06"))):
                x = g[(g.t >= a) & (g.t <= b)].dropna(subset=["gwsa_cm"])
                t = (x.t - pd.Timestamp("2013-01-01")).dt.days.values / 365.25; ww = 2 * np.pi * t
                X = np.column_stack([np.ones_like(t), t, np.sin(ww), np.cos(ww), np.sin(2 * ww), np.cos(2 * ww)])
                gr.append({"unit": u, "window": lab, "trend": np.linalg.lstsq(X, x.gwsa_cm.values, rcond=None)[0][1]})
    G = pd.DataFrame(gr).groupby(["unit", "window"]).trend.agg(["mean", "min", "max"]).unstack("window")
    G.columns = [f"grace_{a}_{b}" for a, b in G.columns]
    T = W[W.season == "pre-monsoon"].merge(G.reset_index(), on="unit")
    T.round(3).to_csv(OUT / "wells_vs_grace.csv", index=False)
    show = T[["unit", "wells", "aquifer", "depth_slope_med_m_yr", "share_falling", "storage_lo_cm_yr", "storage_hi_cm_yr",
              "grace_mean_2013-2020", "grace_min_2013-2020", "grace_max_2013-2020", "grace_mean_2013-2017"]]
    print(show.sort_values("grace_mean_2013-2020").round(2).to_string(index=False))
    from scipy.stats import spearmanr
    mid = (T.storage_lo_cm_yr + T.storage_hi_cm_yr) / 2
    print("\nunits:", len(T), " Spearman(wells mid storage, GRACE 2013-2020) = %.2f" % spearmanr(mid, T["grace_mean_2013-2020"])[0],
          "; with 2013-2017 = %.2f" % spearmanr(mid, T["grace_mean_2013-2017"])[0])
    inside = ((T["grace_mean_2013-2020"] >= T.storage_lo_cm_yr - 0.5) & (T["grace_mean_2013-2020"] <= T.storage_hi_cm_yr + 0.5)).mean()
    print("GRACE within the well-based range (+/-0.5 cm/yr): %.0f%% of units" % (100 * inside))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for _, r in T.iterrows():
        ax.plot([r.storage_lo_cm_yr, r.storage_hi_cm_yr], [r["grace_mean_2013-2020"]] * 2, color="tab:blue" if r.aquifer == "alluvium" else "tab:orange", lw=2)
        ax.plot([(r.storage_lo_cm_yr + r.storage_hi_cm_yr) / 2] * 2, [r["grace_min_2013-2020"], r["grace_max_2013-2020"]], color="0.5", lw=0.8)
        ax.annotate(r.unit.split(" (")[0].split(",")[0], ((r.storage_lo_cm_yr + r.storage_hi_cm_yr) / 2, r["grace_mean_2013-2020"]), fontsize=7)
    lim = [-6, 3]; ax.plot(lim, lim, "k--", lw=0.7); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("wells: storage trend from median May depth trend x specific yield (cm/yr)")
    ax.set_ylabel("GRACE ensemble groundwater trend, 2013-2020 (cm/yr)")
    ax.set_title("CGWB wells vs GRACE by unit (blue alluvium, orange hard rock / mixed)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "wells_vs_grace.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
