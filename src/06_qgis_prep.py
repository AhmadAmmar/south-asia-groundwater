"""Prepare the groundwater results for QGIS (analysis environment; then src/07_qgis_project.py in QGIS Python).

    python src/06_qgis_prep.py

Writes outputs/qgis/:
  groundwater.gpkg   units: the 20 non-overlapping units with ensemble trends per period (cm/yr), 95% half-width,
                     km3/yr, gain and resolvable flag, rain and "rest" trends (step 04), and the CGWB well comparison;
                     wells: CGWB stations with their 2013-2020 May depth trend (m/yr, positive = deeper)
  gwsa_trend_jplcri_era5.tif   per-cell groundwater trend 2002-2026, one member (JPL CRI - ERA5-Land, glacier removed
                     not applied at cell level), cm/yr: for pattern only; units carry the ensemble values
  glacier_cmyr.tif   Hugonnet 2000-2019 glacier mass change as cm water per year over each 0.5 deg cell
  twsa_trend_jplcri.tif  total water storage trend, same period, cm/yr
"""
import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import from_origin

R = Path(__file__).resolve().parents[1]
O = R / "outputs/qgis"; O.mkdir(parents=True, exist_ok=True)
P = R / "data/processed"


def load(name):
    spec = importlib.util.spec_from_file_location(name, R / f"src/{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def tif(path, a, lats, lons):
    """South-to-north 0.5 deg array -> north-up GeoTIFF in EPSG:4326."""
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(lons[0] - 0.25, lats[-1] + 0.25, 0.5, 0.5), nodata=np.nan, compress="deflate") as d:
        d.write(a[::-1].astype("float32"), 1)


def trend_grid(da):
    t = (da.time.to_index() - pd.Timestamp("2002-01-01")).days.values / 365.25
    y = da.values.reshape(len(t), -1); ok = np.isfinite(y).all(0)
    out = np.full(y.shape[1], np.nan); X = np.column_stack([np.ones_like(t), t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    out[ok] = np.linalg.lstsq(X, y[:, ok], rcond=None)[0][1]
    return out.reshape(da.shape[1:])


def main():
    m3 = load("03_ensemble_units")
    a = gpd.read_file(P / "admin1.geojson")
    look = {(c, n): u for u, parts in m3.UNITS.items() for c, n in parts}
    a["unit"] = [look.get((c, n), look.get((c, "*"))) for c, n in zip(a.ADM0_NAME, a.ADM1_NAME)]
    U = a.dropna(subset=["unit"]).dissolve("unit").reset_index()[["unit", "geometry"]]
    T = pd.read_csv(R / "outputs/units_trends.csv")
    wide = T.pivot_table(index="unit", columns="period", values="trend_cm_yr")
    wide.columns = ["trend_" + c.split(" ")[0].replace("-", "_") for c in wide.columns]
    whole = T[T.period == "whole (step at gap)"].set_index("unit")
    wide["ci95_whole"] = 1.96 * whole.total_se; wide["km3_yr_whole"] = whole.km3_yr
    wide["gain"] = whole.gain; wide["resolvable"] = whole.resolvable.astype(str)
    wide["glacier_removed_cm_yr"] = whole.glacier_removed_cm_yr
    cv = pd.read_csv(R / "outputs/climate_vs_rest.csv").groupby("unit").median(numeric_only=True)
    for p in ("2002-2009", "2010-2017", "2018-latest"):
        wide[f"rain_{p[:4]}"] = cv[f"climate_{p}"]; wide[f"rest_{p[:4]}"] = cv[f"rest_{p}"]
    wv = pd.read_csv(R / "outputs/wells_vs_grace.csv").set_index("unit")
    wide["wells_n"] = wv.wells; wide["wells_falling_share"] = wv.share_falling
    wide["wells_storage_lo"] = wv.storage_lo_cm_yr; wide["wells_storage_hi"] = wv.storage_hi_cm_yr
    wide["grace_2013_2020"] = wv["grace_mean_2013-2020"]
    U = U.merge(wide.round(3).reset_index(), on="unit", how="left")
    f = O / "groundwater.gpkg"
    if f.exists():
        f.unlink()
    from shapely.geometry import MultiPolygon
    def polys(g):
        """Polygon parts only (dissolves can leave stray lines), as one MultiPolygon."""
        parts = getattr(g, "geoms", [g])
        return MultiPolygon([q for p in parts for q in (p.geoms if p.geom_type == "MultiPolygon" else [p]) if q.geom_type == "Polygon"])
    U["geometry"] = [polys(g) for g in U.geometry]   # one geometry type for QGIS
    U.to_file(f, layer="units")
    m5 = load("05_wells_check")
    d = pd.read_csv(R / "data/raw/cgwb/cgwb.csv", low_memory=False, parse_dates=["date"]); d = d[(d.currentlevel > 0) & (d.currentlevel < 200)]
    w = m5.well_slopes(d, 5)
    gpd.GeoDataFrame(w, geometry=gpd.points_from_xy(w.lon, w.lat), crs=4326).to_file(f, layer="wells")
    gw = xr.open_dataarray(P / "gwsa_0p5.nc"); tw = xr.open_dataarray(P / "twsa_0p5.nc")
    tif(O / "gwsa_trend_jplcri_era5.tif", trend_grid(gw), gw.lat.values, gw.lon.values)
    tw = tw.assign_coords(time=tw.time.to_index().to_period("M").to_timestamp()).groupby("time").mean()
    tif(O / "twsa_trend_jplcri.tif", trend_grid(tw), tw.lat.values, tw.lon.values)
    g = xr.open_dataarray(P / "glacier_cmyr_0p5.nc"); tif(O / "glacier_cmyr.tif", g.values, g.lat.values, g.lon.values)
    print("units", len(U), "wells", len(w), "written to", O)


if __name__ == "__main__":
    main()
