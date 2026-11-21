"""Fine-scale results for QGIS (step 11; then src/07_qgis_project.py picks them up).

    python src/11_fine_qgis_prep.py

Writes outputs/qgis/fine.gpkg, layer districts: every GAUL level-2 district in the box with
  grace_whole_cm_yr       GRACE groundwater trend 2002-latest (glacier removed), mean over the district: as smooth as GRACE
  downscaled_whole_cm_yr  the step 10 downscaled trend averaged over the district (experimental: see its well test)
  wells, slope_median_m_yr, share_falling, storage_lo/mid/hi_cm_yr   from CGWB wells (step 09; India, >= 5 wells)
  grace_2013_2020_cm_yr   GRACE trend over the well years, for the comparison
  wells_vs_grace          'both losing', 'wells losing, GRACE not', ... where both exist
and GeoTIFFs: gwsa_trend_0p1_downscaled.tif, grace_trend_0p1_plain.tif, wells_krige_slope.tif, wells_krige_sd.tif.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]; P = R / "data/processed"; O = R / "outputs/qgis"; O.mkdir(parents=True, exist_ok=True)
T = from_origin(60, 40, 0.1, 0.1)


def tif(path, a):
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="float32", crs="EPSG:4326",
                       transform=T, nodata=np.nan, compress="deflate") as d:
        d.write(a.astype("float32"), 1)


def main():
    D = xr.open_dataset(P / "gwsa_trend_0p1.nc"); K = xr.open_dataset(P / "wells_krige_0p1.nc")
    tif(O / "gwsa_trend_0p1_downscaled.tif", D.gwsa_trend_whole.values); tif(O / "grace_trend_0p1_plain.tif", D.grace_plain_whole.values)
    tif(O / "wells_krige_slope.tif", K.slope_m_yr.values); tif(O / "wells_krige_sd.tif", K.slope_sd_m_yr.values)
    d = gpd.read_file(P / "districts.geojson")
    ids = rasterize(((g, k + 1) for k, g in enumerate(d.geometry)), out_shape=D.gwsa_trend_whole.shape, transform=T, fill=0, all_touched=False)
    for col, arr in (("grace_whole_cm_yr", D.grace_plain_whole.values), ("downscaled_whole_cm_yr", D.gwsa_trend_whole.values),
                     ("grace_2013_2020_cm_yr", D.grace_plain_2013_2020.values)):
        s = pd.Series(arr.ravel()).groupby(ids.ravel()).mean()
        d[col] = d.index.map(lambda k: s.get(k + 1, np.nan))
    small = d["grace_whole_cm_yr"].isna()                                            # districts smaller than a cell: take the cell at their centre
    c = d[small].geometry.representative_point()
    for col, arr in (("grace_whole_cm_yr", D.grace_plain_whole.values), ("downscaled_whole_cm_yr", D.gwsa_trend_whole.values),
                     ("grace_2013_2020_cm_yr", D.grace_plain_2013_2020.values)):
        d.loc[small, col] = arr[np.clip(((40 - c.y) / .1).astype(int), 0, 349), np.clip(((c.x - 60) / .1).astype(int), 0, 449)]
    w = pd.read_csv(R / "outputs/fine/wells_districts.csv").drop(columns=["ADM0_NAME", "ADM1_NAME", "ADM2_NAME"])
    d = d.merge(w, on="ADM2_CODE", how="left")
    both = d.storage_mid_cm_yr.notna() & d.grace_2013_2020_cm_yr.notna()           # same years as the wells
    lw, lg = d.storage_mid_cm_yr < -0.25, d.grace_2013_2020_cm_yr < -0.25
    d["wells_vs_grace"] = np.where(~both, None, np.select([lw & lg, lw & ~lg, ~lw & lg], ["both losing", "wells losing, GRACE not", "GRACE losing, wells not"], "neither losing"))
    d = d.round(3)
    f = O / "fine.gpkg"
    if f.exists():
        f.unlink()
    d.to_file(f, layer="districts")
    import sqlite3
    with sqlite3.connect(f) as db:
        db.execute("PRAGMA journal_mode=DELETE")
    print(d.wells_vs_grace.value_counts().to_string()); print("districts", len(d), "with wells", int(d.wells.notna().sum()))


if __name__ == "__main__":
    main()
