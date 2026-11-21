"""What goes with falling water tables? District well trends against irrigation, climate and people (step 13).

    python src/13_drivers.py          (after steps 08, 09, 11)

Each predictor of step 08 is averaged over each district (0.1 deg cells whose centre falls in it) and compared with the
district's well-based storage change, 2013-2020 (383 Indian districts with >= 5 wells). Expectations from the hydrology:
  - more dry-season irrigation (green crops in Jan-Apr, high evapotranspiration, cropland) -> faster loss, because in
    north-west India that water is mostly pumped;
  - more rain -> more recharge -> slower loss; a falling rain trend -> faster loss;
  - the strongest loss where irrigation is high AND rain is low (little recharge to offset pumping): tested with a
    two-way table (terciles of dry-season NDVI x terciles of rainfall).
Rank correlations (Spearman) with a bootstrap 95% interval; a district-level random forest with spatially blocked
cross-validation (2 deg blocks) to see how much the predictors explain together.
Writes outputs/fine/district_drivers.csv, outputs/fine/driver_correlations.csv, outputs/fine/irrigation_x_rain.csv and
figures/maps/12_drivers_scatter.png, 13_irrigation_x_rain.png, 14_groundwater_irrigation_map.png.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]; P = R / "data/processed"; FO = R / "outputs/fine"; FIG = R / "figures/maps"
FEATS = {"ndvi_dry": "dry-season greenness (NDVI, Jan-Apr)", "et": "evapotranspiration (mm/yr)", "cropland": "cropland share",
         "precip": "rainfall (mm/yr)", "precip_trend": "rainfall trend (mm/yr per decade)", "lst_dry": "land temperature Mar-May (C)",
         "pop": "population density (log10)", "ndvi_dry_trend": "change in dry-season greenness (per decade)"}
sys.path.insert(0, str(R / "src"))
M = __import__("importlib").import_module


def district_table():
    F = xr.open_dataset(P / "fine_predictors_0p1.nc")
    d = gpd.read_file(R / "outputs/qgis/fine.gpkg", layer="districts")
    ids = rasterize(((g, k + 1) for k, g in enumerate(d.geometry)), out_shape=F.ndvi_dry.shape, transform=from_origin(60, 40, .1, .1), fill=0)
    for k in FEATS:
        s = pd.Series(F[k].values.ravel()).groupby(ids.ravel()).mean()
        d[k] = d.index.map(lambda i: s.get(i + 1, np.nan))
    t = d[d.wells.notna()].dropna(subset=list(FEATS)).copy()
    t["y"] = t.storage_mid_cm_yr
    t.drop(columns="geometry").round(4).to_csv(FO / "district_drivers.csv", index=False)
    return d, t


def correlations(t):
    rng = np.random.default_rng(0); rows = []
    for k, lab in FEATS.items():
        r = spearmanr(t[k], t.y)[0]
        b = [spearmanr(*t[[k, "y"]].sample(len(t), replace=True, random_state=int(rng.integers(1e9))).T.values)[0] for _ in range(1000)]
        rows.append({"predictor": lab, "spearman_with_storage_change": r, "ci_lo": np.percentile(b, 2.5), "ci_hi": np.percentile(b, 97.5)})
    c = pd.DataFrame(rows).sort_values("spearman_with_storage_change"); c.round(3).to_csv(FO / "driver_correlations.csv", index=False)
    print("Rank correlation with district storage change (negative = goes with loss):\n" + c.round(2).to_string(index=False))
    return c


def forest(t):
    from sklearn.ensemble import RandomForestRegressor
    c = t.geometry.representative_point()
    blk = ((c.x // 2).astype(int) * 100 + (c.y // 2).astype(int)).values; ub = np.unique(blk)
    fo = dict(zip(ub, np.random.default_rng(0).integers(0, 5, len(ub)))); f = np.array([fo[b] for b in blk])
    X, y = t[list(FEATS)].values, t.y.values; pred = np.full(len(y), np.nan)
    for k in range(5):
        pred[f == k] = RandomForestRegressor(500, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(X[f != k], y[f != k]).predict(X[f == k])
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2); rho = spearmanr(y, pred)[0]
    print(f"District random forest, 2 deg blocks held out: R2 {r2:.2f}, rank correlation {rho:.2f}")
    return r2, rho


def two_way(t):
    t = t.copy()
    t["irrigation"] = pd.qcut(t.ndvi_dry, 3, labels=["low", "middle", "high"]); t["rain"] = pd.qcut(t.precip, 3, labels=["dry", "middle", "wet"])
    m = t.pivot_table(index="irrigation", columns="rain", values="y", aggfunc="median", observed=False)
    n = t.pivot_table(index="irrigation", columns="rain", values="y", aggfunc="size", observed=False)
    m.round(2).to_csv(FO / "irrigation_x_rain.csv")
    print("Median storage change (cm/yr) by dry-season irrigation x rainfall tercile:\n" + m.round(2).to_string())
    return m, n, t


def figures(t, c, m, n, d, r2, rho):
    import importlib.util
    spec = importlib.util.spec_from_file_location("m12", R / "src/12_maps.py"); m12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m12)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, k in zip(axs, ["cropland", "precip", "et"]):
        ax.scatter(t[k], t.y, s=14, c=t.y, cmap=m12.CMAP, norm=m12.NORM, edgecolor="#555", lw=.2)
        r = c.set_index("predictor").loc[FEATS[k]]
        ax.set_xlabel(FEATS[k]); ax.axhline(0, color="k", lw=.5)
        ax.set_title(f"rank correlation {r.spearman_with_storage_change:+.2f} ({r.ci_lo:+.2f} to {r.ci_hi:+.2f})", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("storage change from wells, cm/yr")
    fig.subplots_adjust(top=0.74, bottom=0.14, wspace=.25)
    m12.frame(fig, "Loss goes with farmland and dry climates; no single factor explains much",
              f"Each dot is an Indian district with 5 or more wells (383), 2013-2020. All eight predictors together, tested on held-out 2 degree blocks,\n"
              f"explain R2 = {r2:.2f} of the differences between districts (rank correlation {rho:.2f}). Evapotranspiration runs the 'wrong' way: MODIS ET is\n"
              f"highest in humid regions and under-reads irrigated dry land, so here it mostly measures wetness. Map 13 shows the combination that matters.",
              m12.SRC_W + "; MODIS NDVI and ET, ERA5-Land rainfall (2003-2024 means)", "12_drivers_scatter.png")
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    im = ax.imshow(m.values.astype(float), cmap=m12.CMAP, norm=m12.NORM)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m.values[i, j]:+.2f}\n({n.values[i, j]} districts)", ha="center", va="center", fontsize=9,
                    color="white" if m.values[i, j] < -2 else "black")
    ax.set_xticks(range(3), [f"{x} rain" for x in m.columns]); ax.set_yticks(range(3), [f"{x} dry-season irrigation" for x in m.index])
    fig.subplots_adjust(top=0.76, left=0.28)
    m12.frame(fig, "Irrigation without rain is where groundwater runs down",
              "Median storage change from wells (cm/yr, 2013-2020) in Indian districts grouped into thirds by dry-season greenness\n"
              "(a sign of irrigation) and by rainfall.", m12.SRC_W + "; MODIS NDVI, ERA5-Land rainfall", "13_irrigation_x_rain.png")
    g = d.to_crs(m12.CRS).copy()
    g["gw_irr"] = (g.ndvi_dry > t.ndvi_dry.quantile(2 / 3)) & (g.precip < t.precip.quantile(1 / 3))
    view = gpd.GeoSeries.from_xy([60, 98], [6, 37], crs=4326).to_crs(m12.CRS).total_bounds
    fig, ax = plt.subplots(figsize=(10, 9)); m12.base(ax, view)
    g[g.gw_irr].plot(ax=ax, color="#cc4c02", edgecolor="white", lw=.2)
    w = g[g.wells.notna()]; w.assign(p=w.geometry.representative_point()).set_geometry("p").plot(
        ax=ax, column="storage_mid_cm_yr", cmap=m12.CMAP, norm=m12.NORM, markersize=12, edgecolor="#333", lw=.3)
    m12.lines(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#cc4c02", label="green in the dry season, little rain\n(likely groundwater-irrigated)")] +
              [Patch(color=cc, label=l) for cc, l in zip(m12.DIV, m12.LAB)], loc="lower left", fontsize=7, title="dots: wells, cm/yr", title_fontsize=8)
    m12.frame(fig, "Where farming most likely runs on groundwater, across borders",
              "Districts in the top third of dry-season greenness and the bottom third of rainfall (thresholds from the Indian districts with wells),\n"
              "in all countries; dots show district well trends in India. Pakistan's Punjab falls in the same class; its well records are not public.",
              "MODIS NDVI 2003-2024, ERA5-Land rainfall; " + m12.SRC_W + "; FAO GAUL 2015 districts", "14_groundwater_irrigation_map.png")


def main():
    d, t = district_table(); c = correlations(t); r2, rho = forest(t); m, n, _ = two_way(t)
    d.to_crs(4326)
    figures(t, c, m, n, d, r2, rho)


if __name__ == "__main__":
    main()
