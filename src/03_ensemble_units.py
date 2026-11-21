"""Groundwater storage change by aquifer / administrative unit, as an ensemble with honest uncertainty (steps 02-03).

    python src/03_ensemble_units.py

What changed from step 01, and why
  1. Ensemble, not one estimate. GWSA = TWSA - (soil + snow + canopy) - glacier, for every combination of
     GRACE solution {JPL mascon CRI, JPL mascon, CSR land, GFZ land} and land model {ERA5-Land, GLDAS Noah, GLDAS CLSM}
     that exists for a month. The spread between members is structural uncertainty that one estimate hides.
  2. Glacier mass removed. Hugonnet et al. (2021) 2000-2019 mean thinning (cm w.e./yr per 0.5 deg cell, from
     src/02_inputs.py glacier) is removed as a linear trend. Acceleration after 2019 is not represented.
  3. Units, not boxes. Non-overlapping groups of admin-1 units (GAUL 2015), named for where water is used and governed.
     A unit smaller than GRACE's ~300 km resolution is still reported, but flagged by its leakage test.
  4. Trends by period. 2002-04..2009-12, 2010-01..2017-06 (GRACE), 2018-06..latest (GRACE-FO), and the whole record
     with a step allowed at the mission gap (an offset between missions would otherwise read as a trend change).
     Each fit has annual and semi-annual harmonics. Standard errors allow for autocorrelated residuals (AR(1) effective
     sample size). Total uncertainty = sqrt(mean within-member SE^2 + between-member SD^2), as a 95% interval.
  5. Leakage test. A uniform -1 cm/yr in one unit, smoothed by a Gaussian of ~300 km full width (GRACE's effective
     resolution), gives the share of the unit's own signal it keeps (gain) and the share of its apparent signal that
     comes from neighbours, under the observed pattern.
Writes outputs/units_trends.csv, outputs/units_members.csv, outputs/units_leakage.csv, outputs/units_monthly.csv,
figures/units_*.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]
P, OUT, FIG = R / "data/processed", R / "outputs", R / "figures"
BOX = (60.0, 5.0, 105.0, 40.0)
LONS = np.arange(BOX[0] + 0.25, BOX[2], 0.5); LATS = np.arange(BOX[1] + 0.25, BOX[3], 0.5)
PERIODS = {"2002-2009": ("2002-04", "2009-12"), "2010-2017": ("2010-01", "2017-06"), "2018-latest": ("2018-06", "2030-01")}
UNITS = {   # unit: list of (country, admin-1) ; "*" = whole country
    "Pakistan Punjab": [("Pakistan", "Punjab"), ("Pakistan", "Islamabad")],
    "Sindh": [("Pakistan", "Sindh")],
    "Balochistan": [("Pakistan", "Balochistan")],
    "Indian Punjab, Haryana, Delhi": [("India", "Punjab"), ("India", "Haryana"), ("India", "Delhi"), ("India", "Chandigarh")],
    "Rajasthan": [("India", "Rajasthan")],
    "Gujarat": [("India", "Gujarat"), ("India", "Daman and Diu"), ("India", "Dadra and Nagar Haveli")],
    "Uttar Pradesh, Uttarakhand": [("India", "Uttar Pradesh"), ("India", "Uttarakhand")],
    "Bihar, Jharkhand": [("India", "Bihar"), ("India", "Jharkhand")],
    "Bengal basin (West Bengal, Bangladesh)": [("India", "West Bengal"), ("Bangladesh", "*")],
    "Madhya Pradesh, Chhattisgarh": [("India", "Madhya Pradesh"), ("India", "Chhattisgarh")],
    "Maharashtra": [("India", "Maharashtra")],
    "Odisha": [("India", "Orissa")],
    "South India (hard rock)": [("India", "Karnataka"), ("India", "Andhra Pradesh"), ("India", "Tamil Nadu"), ("India", "Kerala"),
                                ("India", "Goa"), ("India", "Puducherry")],
    "Northeast India, Bhutan, Sikkim": [("India", x) for x in ("Assam", "Meghalaya", "Arunachal Pradesh", "Nagaland", "Manipur",
                                                              "Mizoram", "Tripura", "Sikkim")] + [("Bhutan", "*"), ("Arunachal Pradesh", "*")],
    "Nepal": [("Nepal", "*")],
    "Western Himalaya, Karakoram": [("Jammu and Kashmir", "*"), ("Aksai Chin", "*"), ("India", "Himachal Pradesh"),
                                    ("Pakistan", "North-West Frontier"), ("Pakistan", "Federally Administered Tribal Areas")],
    "Afghanistan": [("Afghanistan", "*")],
    "Tibet (Xizang)": [("China", "Xizang Zizhiqu")],
    "Qinghai": [("China", "Qinghai Sheng")],
    "Sri Lanka": [("Sri Lanka", "*")],
}


def unit_raster():
    """Integer unit id (1..n, 0 = none) of each 0.5 deg cell, by the admin-1 polygon containing the cell centre."""
    import geopandas as gpd
    from shapely.geometry import Point
    a = gpd.read_file(P / "admin1.geojson")
    look = {}
    for k, (u, parts) in enumerate(UNITS.items(), 1):
        for c, n in parts:
            look[(c, n)] = k
    a["unit"] = [look.get((c, n), look.get((c, "*"), 0)) for c, n in zip(a.ADM0_NAME, a.ADM1_NAME)]
    a = a[a.unit > 0]
    pts = gpd.GeoDataFrame(geometry=[Point(x, y) for y in LATS for x in LONS], crs=4326)
    j = gpd.sjoin(pts, a[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    return j.unit.fillna(0).astype(int).values.reshape(len(LATS), len(LONS))


def to_grid(da):
    """Any (time, lat, lon) DataArray -> monthly on the 0.5 deg box grid (nearest), time as month start."""
    da = da.assign_coords(time=da.time.to_index().to_period("M").to_timestamp()).groupby("time").mean()
    return da.interp(lat=LATS, lon=LONS, method="nearest")


def members():
    """Dictionary (grace, stores) -> GWSA DataArray, glacier trend removed."""
    gr = {"JPL-CRI": xr.open_dataarray(P / "twsa_0p5.nc")}
    for k, f in (("JPL", "twsa_jplnocri_0p5.nc"), ("CSR", "twsa_csr_1deg.nc"), ("GFZ", "twsa_gfz_1deg.nc")):
        if (P / f).exists():
            gr[k] = xr.open_dataarray(P / f)
    gr = {k: to_grid(v) for k, v in gr.items()}
    st = {"ERA5-Land": xr.open_dataset(P / "era5land_stores_anom_0p5.nc")}
    for k, f in (("Noah", "stores_noah_0p5.nc"), ("CLSM", "stores_clsm_0p5.nc")):
        if (P / f).exists():
            st[k] = xr.open_dataset(P / f)
    st = {k: to_grid(v.soil_cm + v.snow_cm + v.canopy_cm) for k, v in st.items()}
    # CSR and GFZ land grids come in metres, smoothed by a 300 km Gaussian, without gain factors: convert to cm and
    # restore amplitude with gains computed as in Landerer & Swenson (2012), from GLDAS Noah storage smoothed the same way
    if "Noah" in st and any(k in gr for k in ("CSR", "GFZ")):
        from scipy.ndimage import gaussian_filter
        x = st["Noah"].fillna(0).values
        xs = np.array([gaussian_filter(m, sigma=300 / 1.1774 / 55.66, mode="nearest") for m in x])   # 300 km half-width
        k = np.clip(np.sum(x * xs, 0) / np.maximum(np.sum(xs * xs, 0), 1e-6), 0.5, 3.0)
        print("gain factors (median, 90th pct):", np.round(np.nanmedian(k), 2), np.round(np.nanpercentile(k, 90), 2), flush=True)
        for c in ("CSR", "GFZ"):
            if c in gr:
                gr[c] = gr[c] * 100 * xr.DataArray(k, dims=("lat", "lon"), coords={"lat": LATS, "lon": LONS})
    gl = xr.open_dataarray(P / "glacier_cmyr_0p5.nc") if (P / "glacier_cmyr_0p5.nc").exists() else None
    out = {}
    for gk, g in gr.items():
        for sk, s in st.items():
            t = sorted(set(g.time.to_index()) & set(s.time.to_index()))
            gw = g.sel(time=t) - s.sel(time=t)
            if gl is not None:
                yrs = xr.DataArray((pd.DatetimeIndex(t) - pd.Timestamp("2006-01-01")).days / 365.25, dims="time", coords={"time": t})
                gw = gw - gl * yrs
            out[(gk, sk)] = gw
    print("members:", list(out), flush=True)
    return out, gr, st, gl


def fit(t, y, step_at=None):
    """OLS trend (cm/yr) with annual and semi-annual harmonics (and optionally a step); AR(1)-adjusted SE."""
    ok = np.isfinite(y); t, y = t[ok], y[ok]
    if len(y) < 24:
        return np.nan, np.nan, np.nan
    w = 2 * np.pi * t
    cols = [np.ones_like(t), t - t.mean(), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)]
    if step_at is not None and 12 <= (t > step_at).sum() <= len(t) - 12:   # a step needs data on both sides
        cols.append((t > step_at).astype(float))
    else:
        step_at = None
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    r1 = np.corrcoef(r[:-1], r[1:])[0, 1] if len(r) > 3 else 0
    neff = max(len(y) * (1 - r1) / (1 + r1), 8)
    s2 = np.sum(r ** 2) / (neff - X.shape[1])
    cov = s2 * np.linalg.pinv(X.T @ X)
    return b[1], np.sqrt(cov[1, 1]), (b[-1] if step_at is not None else np.nan)


def leakage(ur, n):
    """Gain and neighbour share per unit from a ~300 km Gaussian smoothing of unit-by-unit unit signals."""
    from scipy.ndimage import gaussian_filter
    area = np.cos(np.radians(LATS))[:, None] * np.ones_like(ur, float)
    L = np.zeros((n, n))
    for v in range(1, n + 1):
        sm = gaussian_filter((ur == v).astype(float), sigma=2.5, mode="constant")   # sigma 1.25 deg, FWHM ~ 320 km
        for u in range(1, n + 1):
            m = ur == u
            L[u - 1, v - 1] = np.sum(sm[m] * area[m]) / np.sum(area[m])
    return L


def main():
    ur = unit_raster(); names = list(UNITS); n = len(names)
    area_km2 = np.cos(np.radians(LATS))[:, None] * (0.5 * 111.32) ** 2 * np.ones_like(ur, float)
    mem, gr, st, gl = members()
    rows, mrows = [], []
    for (gk, sk), gw in mem.items():
        a = gw.values; tt = gw.time.to_index()
        for u in range(1, n + 1):
            m = ur == u
            w = area_km2[m]
            ser = np.array([np.nansum(a[i][m] * w) / np.sum(w * np.isfinite(a[i][m])) if np.isfinite(a[i][m]).any() else np.nan for i in range(len(tt))])
            for i, t in enumerate(tt):
                mrows.append({"unit": names[u - 1], "grace": gk, "stores": sk, "month": t.strftime("%Y-%m"), "gwsa_cm": ser[i]})
            yrs = (tt - pd.Timestamp("2002-01-01")).days.values / 365.25
            for pn, (a0, a1) in PERIODS.items():
                sel = (tt >= a0) & (tt <= a1)
                if sel.sum() < 36:
                    continue
                s, se, _ = fit(yrs[sel], ser[sel])
                rows.append({"unit": names[u - 1], "grace": gk, "stores": sk, "period": pn, "trend": s, "se": se})
            s, se, step = fit(yrs, ser, step_at=(pd.Timestamp("2018-01-01") - pd.Timestamp("2002-01-01")).days / 365.25)
            rows.append({"unit": names[u - 1], "grace": gk, "stores": sk, "period": "whole (step at gap)", "trend": s, "se": se, "step_cm": step})
    M = pd.DataFrame(rows); M.to_csv(OUT / "units_members.csv", index=False)
    # CLSM assimilates GRACE, so part of GRACE's groundwater loss is already in its soil store: subtracting it removes
    # real depletion. It is kept as a sensitivity (units_members.csv, column clsm_trend below) but not in the ensemble.
    clsm = M[(M.stores == "CLSM") & (M.period == "whole (step at gap)")].groupby("unit").trend.mean()
    M = M[M.stores != "CLSM"]
    pd.DataFrame(mrows).to_csv(OUT / "units_monthly.csv", index=False)
    g = M.groupby(["unit", "period"])
    T = pd.DataFrame({"trend_cm_yr": g.trend.mean(), "within_se": g.se.apply(lambda s: np.sqrt(np.mean(s ** 2))),
                      "between_sd": g.trend.std(ddof=1), "members": g.trend.count(), "step_cm": g.step_cm.mean() if "step_cm" in M else np.nan}).reset_index()
    T["clsm_sensitivity"] = np.where(T.period == "whole (step at gap)", T.unit.map(clsm), np.nan)
    T["total_se"] = np.sqrt(T.within_se ** 2 + T.between_sd.fillna(0) ** 2)
    T["lo95"] = T.trend_cm_yr - 1.96 * T.total_se; T["hi95"] = T.trend_cm_yr + 1.96 * T.total_se
    ua = {names[u - 1]: float(area_km2[ur == u].sum()) for u in range(1, n + 1)}
    T["area_km2"] = T.unit.map(ua); T["km3_yr"] = T.trend_cm_yr * T.area_km2 * 1e-5
    T["glacier_removed_cm_yr"] = T.unit.map({names[u - 1]: float(np.sum((gl.values if gl is not None else 0 * ur)[ur == u] * area_km2[ur == u]) / ua[names[u - 1]])
                                            for u in range(1, n + 1)})
    L = leakage(ur, n)
    lk = pd.DataFrame({"unit": names, "gain": np.diag(L)})
    obs = T[T.period == "whole (step at gap)"].set_index("unit").trend_cm_yr.reindex(names).values
    with np.errstate(invalid="ignore", divide="ignore"):
        own = np.diag(L) * obs; tot = L @ np.nan_to_num(obs)
        lk["neighbour_share"] = np.abs(tot - own) / (np.abs(own) + np.abs(tot - own))
    lk["resolvable"] = (lk.gain >= 0.5) & (lk.neighbour_share <= 0.5)
    lk.to_csv(OUT / "units_leakage.csv", index=False)
    T = T.merge(lk, on="unit"); T.round(3).to_csv(OUT / "units_trends.csv", index=False)
    piv = T.pivot_table(index="unit", columns="period", values="trend_cm_yr").round(2)
    piv["+/-95 (whole)"] = (1.96 * T[T.period == "whole (step at gap)"].set_index("unit").total_se).round(2)
    piv["km3/yr (whole)"] = T[T.period == "whole (step at gap)"].set_index("unit").km3_yr.round(1)
    piv["step cm"] = T[T.period == "whole (step at gap)"].set_index("unit").step_cm.round(1)
    piv["gain"] = lk.set_index("unit").gain.round(2); piv["resolvable"] = lk.set_index("unit").resolvable
    print(piv.sort_values("km3/yr (whole)").to_string())
    figures(T, names)


def figures(T, names):
    """Period trends with 95% intervals per unit, and the ensemble series of the main plains units."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = list(PERIODS) + ["whole (step at gap)"]
    order = T[T.period == per[-1]].sort_values("trend_cm_yr").unit.tolist()
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 1.5))
    for k, p in enumerate(per):
        d = T[T.period == p].set_index("unit").reindex(order)
        y = np.arange(len(order)) + (k - 1.5) * 0.18
        ax.errorbar(d.trend_cm_yr, y, xerr=1.96 * d.total_se, fmt="o", ms=3.5, lw=1, label=p)
    ax.axvline(0, color="k", lw=0.6); ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("groundwater storage trend, cm of water per year (95% interval: within + between members)")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(FIG / "units_trends.png", dpi=150); plt.close(fig)
    mo = pd.read_csv(OUT / "units_monthly.csv"); mo["t"] = pd.to_datetime(mo.month)
    sel = ["Indian Punjab, Haryana, Delhi", "Rajasthan", "Pakistan Punjab", "Uttar Pradesh, Uttarakhand", "Bihar, Jharkhand",
           "Bengal basin (West Bengal, Bangladesh)", "South India (hard rock)", "Western Himalaya, Karakoram"]
    fig, axs = plt.subplots(len(sel), 1, figsize=(10, 1.9 * len(sel)), sharex=True)
    for ax, u in zip(axs, sel):
        d = mo[mo.unit == u]
        for (gk, sk), m in d.groupby(["grace", "stores"]):
            ax.plot(m.t, m.gwsa_cm, lw=0.5, alpha=0.6)
        e = d.groupby("t").gwsa_cm.agg(["mean"]).rolling(12, center=True, min_periods=6).mean()
        ax.plot(e.index, e["mean"], "k", lw=1.6)
        ax.axvspan(pd.Timestamp("2017-07-01"), pd.Timestamp("2018-05-31"), color="0.85")
        ax.set_title(u, fontsize=9, loc="left"); ax.set_ylabel("cm")
    fig.tight_layout(); fig.savefig(FIG / "units_series.png", dpi=150); plt.close(fig)
    print("figures written", flush=True)


if __name__ == "__main__":
    main()
