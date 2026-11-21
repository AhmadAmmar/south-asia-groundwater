"""Downscaling the GRACE groundwater trend from 0.5 deg to 0.1 deg (step 10), conserving GRACE's mass, tested on wells.

    python src/10_downscale.py

1. Target: the groundwater-storage trend of each 0.5 deg cell (JPL CRI mascons minus ERA5-Land soil, snow and canopy,
   step 01; trend with an annual cycle, cm/yr), with the step 02 glacier loss removed. Two periods: the whole record
   (2002-04 to the latest month) for the maps, and 2013-2020 to match the CGWB well trends for the test.
2. Model: random forest on the step 08 predictors averaged to 0.5 deg (5 x 5 fine cells -> one GRACE cell). GRACE
   itself is not a predictor (that would copy the answer). Skill at 0.5 deg: 5-fold cross-validation with 5 deg spatial
   blocks held out.
3. Fine prediction: the same model on the 0.1 deg predictors, then a correction per 0.5 deg cell so that the mean of its
   25 fine cells equals the GRACE value exactly: the downscaling moves water around inside a cell but never creates or
   removes it. (Known limit: a model trained at 0.5 deg is applied at 0.1 deg, where relations can differ.)
4. Test on wells (never used in fitting): for 0.1 deg cells and for districts with wells, rank correlation between the
   2013-2020 downscaled trend and the well trend (sign flipped: falling water table = losing storage), compared with the
   same correlation for plain GRACE (the 0.5 deg value spread evenly), and inside GRACE cells (the added detail itself).
   Downscaling earns its place only if it beats that. A second model learns the within-cell pattern from the wells.
Writes data/processed/gwsa_trend_0p1.nc, outputs/fine/downscale_skill.csv, outputs/fine/downscale_importance.csv.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]; P = R / "data/processed"; OUT = R / "outputs/fine"; OUT.mkdir(parents=True, exist_ok=True)
FEATS = ["ndvi_dry", "ndvi_dry_trend", "cropland", "et", "precip", "precip_trend", "lst_dry", "pop", "elev", "slope"]
PERIODS = {"whole": (None, None), "2013_2020": ("2013-01-01", "2020-12-31")}


def load(name):
    spec = importlib.util.spec_from_file_location(name, R / f"src/{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def coarse_target(period):
    """0.5 deg trend (cm/yr) north-to-south, glacier loss removed."""
    m6 = load("06_qgis_prep")
    gw = xr.open_dataarray(P / "gwsa_0p5.nc"); a, b = PERIODS[period]
    if a:
        gw = gw.sel(time=slice(a, b))
    t = m6.trend_grid(gw)                                       # south-to-north, like the file
    g = xr.open_dataarray(P / "glacier_cmyr_0p5.nc").values
    t = t - np.nan_to_num(g)
    return t[::-1]


def to_coarse(fine):
    """Mean of each 5 x 5 block (0.1 -> 0.5 deg), north-to-south."""
    ny, nx = fine.shape
    return np.nanmean(fine.reshape(ny // 5, 5, nx // 5, 5), axis=(1, 3))


def table(F, coarse=False):
    arr = {k: (to_coarse(F[k].values) if coarse else F[k].values) for k in FEATS}
    return pd.DataFrame({k: v.ravel() for k, v in arr.items()}), arr[FEATS[0]].shape


def fit(F, target):
    X, shp = table(F, coarse=True); y = target.ravel()
    ok = np.isfinite(y) & X.notna().all(1).values
    rows, cols = np.unravel_index(np.arange(y.size), shp)
    block = (rows // 10) * 100 + cols // 10                      # 5 deg blocks
    ub = np.unique(block[ok]); rng = np.random.default_rng(0); fold_of = dict(zip(ub, rng.integers(0, 5, len(ub))))
    fold = np.array([fold_of.get(b, -1) for b in block]); pred = np.full(y.size, np.nan)
    for k in range(5):
        tr, te = ok & (fold != k), ok & (fold == k)
        m = RandomForestRegressor(400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    r2 = 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
    model = RandomForestRegressor(400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X[ok], y[ok])
    return model, r2, ok.sum()


def downscale(F, model, target):
    X, shp = table(F); ok = X.notna().all(1).values
    p = np.full(len(X), np.nan); p[ok] = model.predict(X[ok]); p = p.reshape(shp)
    mean = to_coarse(p)
    corr = np.kron(target - mean, np.ones((5, 5)))               # same correction for all 25 fine cells of a GRACE cell
    out = p + corr
    out[~np.isfinite(np.kron(target, np.ones((5, 5))))] = np.nan
    return out


def well_test(fine, plain, F):
    import geopandas as gpd
    m9 = load("09_wells_fine")
    w = m9.well_trends(); c = m9.binned(w)
    lats, lons = F.lat.values, F.lon.values
    i = np.clip(np.floor((40 - c.lat) / 0.1).astype(int), 0, len(lats) - 1); j = np.clip(np.floor((c.lon - 60) / 0.1).astype(int), 0, len(lons) - 1)
    c["down"] = fine[i, j]; c["plain"] = plain[i, j]; c["pi"] = i // 5; c["pj"] = j // 5; c = c.dropna(subset=["down", "plain"])
    rows = [{"level": "0.1 deg cells with wells", "n": len(c), "spearman_downscaled": spearmanr(-c.slope, c.down)[0],
             "spearman_plain_grace": spearmanr(-c.slope, c.plain)[0]}]
    # The direct test of the added detail: inside each GRACE cell, do fine cells the model calls worse have faster-falling wells?
    c["parent"] = c.pi.astype(str) + "_" + c.pj.astype(str)
    g = c[c.groupby("parent").slope.transform("size") >= 4].copy()
    g["w_anom"] = -(g.slope - g.groupby("parent").slope.transform("mean")); g["d_anom"] = g.down - g.groupby("parent").down.transform("mean")
    rows.append({"level": "within GRACE cells (anomalies, cells with >= 4 well cells)", "n": len(g),
                 "spearman_downscaled": spearmanr(g.w_anom, g.d_anom)[0], "spearman_plain_grace": 0.0})
    d = gpd.read_file(P / "districts.geojson"); wd = pd.read_csv(OUT / "wells_districts.csv")
    pts = gpd.GeoDataFrame(c, geometry=gpd.points_from_xy(c.lon, c.lat), crs=4326)
    j2 = gpd.sjoin(pts, d[["ADM2_CODE", "geometry"]], predicate="within").groupby("ADM2_CODE")[["down", "plain"]].mean().reset_index()
    j2 = j2.merge(wd[["ADM2_CODE", "slope_median_m_yr", "storage_mid_cm_yr"]], on="ADM2_CODE")
    rows.append({"level": "districts with >= 5 wells", "n": len(j2), "spearman_downscaled": spearmanr(j2.storage_mid_cm_yr, j2.down)[0],
                 "spearman_plain_grace": spearmanr(j2.storage_mid_cm_yr, j2.plain)[0]})
    return pd.DataFrame(rows)


def wells_trained_within(F):
    """Second attempt at the detail: learn the within-GRACE-cell pattern from the wells themselves (well anomaly about
    its 0.5 deg cell mean vs predictor anomalies), 5-fold with 2 deg blocks held out. Returns the held-out rank correlation."""
    m9 = load("09_wells_fine")
    c = m9.binned(m9.well_trends())
    i = np.floor((40 - c.lat) / .1).astype(int).values; j = np.floor((c.lon - 60) / .1).astype(int).values
    par = pd.Series((i // 5).astype(str)) + "_" + pd.Series((j // 5).astype(str))
    XA = pd.DataFrame({k: F[k].values[i, j] - np.kron(to_coarse(F[k].values), np.ones((5, 5)))[i, j] for k in FEATS})
    y = -(c.slope.values - c.groupby(par.values).slope.transform("mean").values)
    ok = XA.notna().all(1).values & (c.groupby(par.values).slope.transform("size").values >= 4)
    blk = ((c.lon // 2).astype(int) * 100 + (c.lat // 2).astype(int)).values; ub = np.unique(blk)
    fo = dict(zip(ub, np.random.default_rng(0).integers(0, 5, len(ub)))); f = np.array([fo[b] for b in blk]); pred = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = ok & (f != k), ok & (f == k)
        pred[te] = RandomForestRegressor(300, min_samples_leaf=20, n_jobs=-1, random_state=0).fit(XA[tr], y[tr]).predict(XA[te])
    return {"level": "within GRACE cells, model trained on wells (2 deg blocks held out)", "n": int(ok.sum()),
            "spearman_downscaled": spearmanr(y[ok], pred[ok])[0], "spearman_plain_grace": 0.0}


def main():
    F = xr.open_dataset(P / "fine_predictors_0p1.nc")
    F["ndvi_dry_trend"] = F.ndvi_dry_trend.clip(-0.5, 0.5); F["precip_trend"] = F.precip_trend.clip(-400, 400)   # sensor-edge outliers
    out, skill = {}, []
    for period in PERIODS:
        tgt = coarse_target(period)
        model, r2, n = fit(F, tgt)
        print(f"{period}: random forest at 0.5 deg, 5 deg blocks held out: R2 = {r2:.2f} ({n} cells)")
        out[period] = downscale(F, model, tgt)
        out[period + "_plain"] = np.kron(tgt, np.ones((5, 5)))
        skill.append({"period": period, "cells_0p5": n, "cv_r2_0p5": r2})
        if period == "whole":
            pd.DataFrame({"predictor": FEATS, "importance": model.feature_importances_}).sort_values("importance", ascending=False).round(3).to_csv(OUT / "downscale_importance.csv", index=False)
    wt = well_test(out["2013_2020"], out["2013_2020_plain"], F)
    wt = pd.concat([wt, pd.DataFrame([wells_trained_within(F)])], ignore_index=True)
    print("test on CGWB wells, 2013-2020 (rank correlation with storage change from wells):\n" + wt.round(3).to_string(index=False))
    pd.concat([pd.DataFrame(skill), wt], ignore_index=True).round(3).to_csv(OUT / "downscale_skill.csv", index=False)
    xr.Dataset({"gwsa_trend_whole": (("lat", "lon"), out["whole"].astype("float32")),
                "gwsa_trend_2013_2020": (("lat", "lon"), out["2013_2020"].astype("float32")),
                "grace_plain_whole": (("lat", "lon"), out["whole_plain"].astype("float32")),
                "grace_plain_2013_2020": (("lat", "lon"), out["2013_2020_plain"].astype("float32"))},
               coords={"lat": F.lat, "lon": F.lon}, attrs={"units": "cm water per year", "method": "src/10_downscale.py"}).to_netcdf(P / "gwsa_trend_0p1.nc")
    print(pd.read_csv(OUT / "downscale_importance.csv").to_string(index=False))


if __name__ == "__main__":
    main()
