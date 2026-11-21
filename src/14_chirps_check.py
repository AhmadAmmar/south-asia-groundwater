"""Where does CHIRPS rainfall jump on its own? A consistency map against ERA5-Land (step 14).

    python src/14_chirps_check.py

For each 0.25 deg cell of 60-105 E, 5-40 N: mean annual rainfall in 2010-2024 divided by 2003-2008, in CHIRPS (pentad)
and in ERA5-Land (monthly). Real climate shifts appear in both; a station-input artefact in only one. The map shows
the ratio of the two ratios (CHIRPS change / ERA5-Land change): near 1 = consistent; well above 1 = CHIRPS rose on its
own. Annual time series at a few places show what the jump looks like.
Writes data/processed/chirps_vs_era5_0p25.nc, outputs/chirps_check_points.csv, figures/maps/15_chirps_artefact.png.
"""
import io
import sys
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]; P = R / "data/processed"; FIG = R / "figures/maps"
BOX = (60.0, 5.0, 105.0, 40.0); RES = 0.25; NX, NY = 180, 140
PLACES = {"Ludhiana (Indian Punjab)": (75.85, 30.9), "Lahore (Pakistan Punjab)": (74.35, 31.55), "Hisar (Haryana)": (75.7, 29.15),
          "Jaipur (Rajasthan)": (75.8, 26.9), "Patna (Bihar)": (85.1, 25.6), "Dhaka (Bangladesh)": (90.4, 23.8)}


def annual(col, band, scale):
    return ee.ImageCollection(ee.List.sequence(2003, 2024).map(lambda y: ee.ImageCollection(col).filter(ee.Filter.calendarRange(y, y, "year"))
                                                               .select(band).sum().multiply(scale).set("year", y)))


def grid(img):
    import rasterio
    url = img.toFloat().unmask(-9999).getDownloadURL({"crs": "EPSG:4326", "crs_transform": [RES, 0, BOX[0], 0, -RES, BOX[3]], "dimensions": f"{NX}x{NY}", "format": "GEO_TIFF"})
    with rasterio.open(io.BytesIO(requests.get(url, timeout=900).content)) as s:
        a = s.read(1)
    a[a <= -9998] = np.nan
    return a


def main():
    ee.Initialize(project="ee-ahmadammargeo")
    ch, er = annual("UCSB-CHG/CHIRPS/PENTAD", "precipitation", 1), annual("ECMWF/ERA5_LAND/MONTHLY_AGGR", "total_precipitation_sum", 1000)
    def ratio(c):
        post = c.filter(ee.Filter.gte("year", 2010)).mean(); pre = c.filter(ee.Filter.lte("year", 2008)).mean()
        return post.divide(pre.max(50))
    f = P / "chirps_vs_era5_0p25.nc"
    if not f.exists():
        rc, re_ = grid(ratio(ch)), grid(ratio(er))
        lats = BOX[3] - RES / 2 - RES * np.arange(NY); lons = BOX[0] + RES / 2 + RES * np.arange(NX)
        xr.Dataset({"chirps_ratio": (("lat", "lon"), rc), "era5_ratio": (("lat", "lon"), re_)}, coords={"lat": lats, "lon": lons}).to_netcdf(f)
    d = xr.open_dataset(f); rr = d.chirps_ratio / d.era5_ratio
    rows = []
    for n, (x, y) in PLACES.items():
        pt = ee.Geometry.Point(x, y)
        for name, c in (("CHIRPS", ch), ("ERA5-Land", er)):
            v = c.map(lambda i: ee.Feature(None, {"year": i.get("year"), "mm": i.reduceRegion(ee.Reducer.mean(), pt, 5000).values().get(0)})).getInfo()["features"]
            rows += [{"place": n, "product": name, "year": int(q["properties"]["year"]), "mm": q["properties"]["mm"]} for q in v]
    t = pd.DataFrame(rows); t.to_csv(R / "outputs/chirps_check_points.csv", index=False)
    import importlib.util
    spec = importlib.util.spec_from_file_location("m12", R / "src/12_maps.py"); m12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m12)
    fig = plt.figure(figsize=(15, 8.4)); ax = fig.add_axes([0.01, 0.08, 0.5, 0.78]); m12.base(ax)
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cols = ["#2166ac", "#92c5de", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#67001f"]; br = [0, 0.8, 0.9, 1.1, 1.25, 1.5, 2, 9]
    m = m12.grid_plot(ax, rr.values, (BOX[0], BOX[2], BOX[1], BOX[3]), ListedColormap(cols), BoundaryNorm(br, len(cols))); m12.lines(ax)
    fig.colorbar(m, ax=ax, shrink=.6, label="CHIRPS change / ERA5-Land change (1 = consistent)")
    for n, (x, y) in PLACES.items():
        X, Y = m12.gpd.GeoSeries.from_xy([x], [y], crs=4326).to_crs(m12.CRS).iloc[0].coords[0]; ax.plot(X, Y, "ko", ms=3); ax.annotate(n.split(" (")[0], (X, Y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    for k, n in enumerate(PLACES):
        a = fig.add_axes([0.58 + (k % 2) * 0.21, 0.66 - (k // 2) * 0.27, 0.18, 0.19])
        for prod, c in (("CHIRPS", "#d6604d"), ("ERA5-Land", "#2166ac")):
            s = t[(t.place == n) & (t["product"] == prod)]; a.plot(s.year, s.mm, color=c, lw=1.4, label=prod)
        a.set_title(n, fontsize=8); a.tick_params(labelsize=7)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        if k == 0:
            a.legend(fontsize=7, frameon=False); a.set_ylabel("mm/yr", fontsize=7)
    m12.frame(fig, "A rainfall product that jumps: check CHIRPS before trusting its trends",
              "Left: how much CHIRPS rainfall changed from 2003-2008 to 2010-2024, divided by the change in ERA5-Land. Red: CHIRPS rose without ERA5-Land.\n"
              "Right: annual rainfall at six places. Over Indian Punjab CHIRPS nearly triples around 2009-2010 (Ludhiana x2.8 vs x1.1); weaker in Haryana, Balochistan and Sindh.",
              "CHIRPS v2.0 pentad (UCSB-CHG); ERA5-Land monthly aggregates (ECMWF/Copernicus)", "15_chirps_artefact.png")
    w = (rr > 1.5).sum().item(); print("cells where CHIRPS rose > 1.5x more than ERA5-Land:", w, "of", int(np.isfinite(rr).sum()))
    print(t.pivot_table(index="place", columns="product", values="mm", aggfunc=lambda s: s.iloc[7:].mean() / s.iloc[:6].mean()).round(2))


if __name__ == "__main__":
    main()
