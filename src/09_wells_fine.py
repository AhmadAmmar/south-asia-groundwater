"""Wells at fine scale (step 09): CGWB well trends by district, and a kriged surface with its uncertainty.

    python src/09_wells_fine.py

Wells: CGWB May (pre-monsoon) depth to water, 2013-2020, slope per station (m/yr, positive = water table falling), as in
step 05. Slopes beyond +-5 m/yr (a few hundred stations, mostly gauge resets or pumping-well readings) are dropped.

Districts (FAO GAUL 2015 level 2): number of wells, median slope and interquartile range, share falling, and storage change
= -median x specific yield, with Sy from the step 05 aquifer classes (alluvium 0.08-0.16, hard rock 0.01-0.03), giving a
low-high range in cm of water per year. Districts with fewer than 5 wells get no value.

Surface: wells are first binned to 0.1 deg cells (median per cell, so one dense town does not dominate), then ordinary
kriging (exponential variogram, 40 nearest cells), on the step 08 grid. Cells more than 30 km from any well cell are left
empty. Skill is checked by 5-fold cross-validation with 2 deg spatial blocks held out (a block's wells are never used to
predict it), against the simplest alternative (the mean of the training wells).

Writes outputs/fine/wells_districts.csv, outputs/fine/wells_krige_cv.csv, data/processed/wells_krige_0p1.nc.
"""
import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]; P = R / "data/processed"; OUT = R / "outputs/fine"; OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    spec = importlib.util.spec_from_file_location(name, R / f"src/{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def well_trends():
    m5 = load("05_wells_check")
    d = pd.read_csv(R / "data/raw/cgwb/cgwb.csv", low_memory=False, parse_dates=["date"]); d = d[(d.currentlevel > 0) & (d.currentlevel < 200)]
    w = m5.well_slopes(d, 5)
    n0 = len(w); w = w[w.slope.abs() <= 5].reset_index(drop=True)
    print(f"wells with a 2013-2020 May trend: {n0:,}; kept {len(w):,} (|slope| <= 5 m/yr)")
    return gpd.GeoDataFrame(w, geometry=gpd.points_from_xy(w.lon, w.lat), crs=4326)


def aquifer_of(dist):
    """Step 05 aquifer class (alluvium or hard rock) of each district, from the unit its centroid falls in."""
    m3 = load("03_ensemble_units"); m5 = load("05_wells_check")
    ur = m3.unit_raster(); names = list(m3.UNITS)
    c = dist.geometry.representative_point()
    i = ((c.y - 5) / 0.5).astype(int).clip(0, ur.shape[0] - 1); j = ((c.x - 60) / 0.5).astype(int).clip(0, ur.shape[1] - 1)
    unit = [names[u - 1] if u > 0 else None for u in ur[i.values, j.values]]
    return pd.Series(["hard" if u in m5.HARD else "alluvium" for u in unit], index=dist.index), pd.Series(unit, index=dist.index)


def by_district(w):
    m5 = load("05_wells_check")
    dist = gpd.read_file(P / "districts.geojson")
    j = gpd.sjoin(w, dist[["ADM2_CODE", "geometry"]], predicate="within")
    g = j.groupby("ADM2_CODE").slope
    t = pd.DataFrame({"wells": g.size(), "slope_median_m_yr": g.median(), "slope_q25": g.quantile(.25), "slope_q75": g.quantile(.75),
                      "share_falling": g.apply(lambda s: (s > 0.1).mean())}).reset_index()
    dist["aquifer"], dist["unit"] = aquifer_of(dist)
    t = dist[["ADM0_NAME", "ADM1_NAME", "ADM2_NAME", "ADM2_CODE", "aquifer", "unit"]].merge(t, on="ADM2_CODE", how="inner")
    t = t[t.wells >= 5].copy()
    sy = np.where(t.aquifer == "hard", 0.01, 0.08), np.where(t.aquifer == "hard", 0.03, 0.16)
    t["storage_lo_cm_yr"] = -t.slope_median_m_yr * np.where(t.slope_median_m_yr > 0, sy[1], sy[0]) * 100
    t["storage_hi_cm_yr"] = -t.slope_median_m_yr * np.where(t.slope_median_m_yr > 0, sy[0], sy[1]) * 100
    t["storage_mid_cm_yr"] = (t.storage_lo_cm_yr + t.storage_hi_cm_yr) / 2
    t.round(4).to_csv(OUT / "wells_districts.csv", index=False)
    print(f"districts with >= 5 wells: {len(t)}; median wells per district {int(t.wells.median())}")
    return t


def binned(w, res=0.1):
    b = w.assign(ci=((w.lon - 60) // res).astype(int), cj=((w.lat - 5) // res).astype(int))
    c = b.groupby(["ci", "cj"]).agg(slope=("slope", "median"), n=("slope", "size")).reset_index()
    c["lon"] = 60 + (c.ci + .5) * res; c["lat"] = 5 + (c.cj + .5) * res
    return c


def krige(c, lons, lats, n_closest=40):
    from pykrige.ok import OrdinaryKriging
    ok = OrdinaryKriging(c.lon.values, c.lat.values, c.slope.values, variogram_model="exponential", nlags=30, coordinates_type="geographic")
    z, ss = ok.execute("grid", lons, lats, backend="C", n_closest_points=n_closest)
    return np.asarray(z), np.sqrt(np.asarray(ss)), ok


def cross_validate(c, n_closest=40):
    """Two tests: 'blocked' holds out whole 2 deg blocks (can the surface reach far from wells?); 'random' holds out random
    cells with their neighbours still in (how good is the surface inside the network, where it is drawn?)."""
    res = pd.DataFrame([{"test": mode, **_cv(c, mode, n_closest)} for mode in ("blocked 2 deg", "random cells")])
    res.round(3).to_csv(OUT / "wells_krige_cv.csv", index=False)
    print("kriging cross-validation (5 folds):\n" + res.round(3).to_string(index=False))
    return res


def _cv(c, mode, n_closest):
    from pykrige.ok import OrdinaryKriging
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    if mode.startswith("blocked"):
        block = ((c.lon // 2).astype(int).astype(str) + "_" + (c.lat // 2).astype(int).astype(str))
        ub = block.unique(); fold = dict(zip(ub, rng.integers(0, 5, len(ub)))); c = c.assign(fold=block.map(fold))
    else:
        c = c.assign(fold=rng.integers(0, 5, len(c)))
    pred = np.full(len(c), np.nan)
    for k in range(5):
        tr, te = c[c.fold != k], c[c.fold == k]
        ok = OrdinaryKriging(tr.lon.values, tr.lat.values, tr.slope.values, variogram_model="exponential", nlags=30, coordinates_type="geographic")
        z, _ = ok.execute("points", te.lon.values, te.lat.values, backend="C", n_closest_points=n_closest)
        pred[te.index] = z
    y = c.slope.values; base = np.array([c[c.fold != f].slope.mean() for f in c.fold])
    return {"cells": len(c), "rmse_krige_m_yr": np.sqrt(np.nanmean((y - pred) ** 2)), "rmse_mean_m_yr": np.sqrt(np.mean((y - base) ** 2)),
            "skill_vs_mean": 1 - np.nansum((y - pred) ** 2) / np.nansum((y - base) ** 2),
            "spearman": spearmanr(y, pred, nan_policy="omit")[0], "sign_agreement": np.mean(np.sign(y) == np.sign(pred))}


def main():
    w = well_trends(); by_district(w)
    c = binned(w); print(f"0.1 deg cells with wells: {len(c):,}")
    cross_validate(c)
    f = xr.open_dataset(P / "fine_predictors_0p1.nc"); lons, lats = f.lon.values, f.lat.values
    z, sd, _ = krige(c, lons, lats[::-1])
    z, sd = z[::-1], sd[::-1]                                          # back to north-to-south
    from scipy.spatial import cKDTree
    dist_km, _ = cKDTree(np.c_[c.lon, c.lat]).query(np.c_[np.repeat(lons[None], len(lats), 0).ravel(), np.repeat(lats[:, None], len(lons), 1).ravel()])
    far = (dist_km.reshape(len(lats), len(lons)) * 111) > 30
    z[far] = np.nan; sd[far] = np.nan
    xr.Dataset({"slope_m_yr": (("lat", "lon"), z.astype("float32")), "slope_sd_m_yr": (("lat", "lon"), sd.astype("float32"))},
               coords={"lat": lats, "lon": lons}).to_netcdf(P / "wells_krige_0p1.nc")
    print("kriged cells:", int(np.isfinite(z).sum()))


if __name__ == "__main__":
    main()
