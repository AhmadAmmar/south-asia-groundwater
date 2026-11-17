"""Groundwater storage change over South Asia and the greater Third Pole from GRACE and GRACE-FO, 2002 to the latest month.

    python src/01_grace_groundwater.py

1. Total water storage anomaly (TWSA): NASA JPL GRACE/GRACE-FO mascons RL06.3 v04 with the coastline resolution
   improvement (CRI), 0.5 deg grid of 3 deg equal-area mascons (true resolution ~300 km), cm of water, anomaly about
   the 2004.0-2009.999 mean; downloaded from PO.DAAC with earthaccess (Earthdata login in ~/_netrc). The JPL land
   scaling factors are applied (they restore signal lost to the 3 deg mascons on land).
2. The other stores, from ERA5-Land monthly (Earth Engine ECMWF/ERA5_LAND/MONTHLY_AGGR), as anomalies over the same
   baseline: soil moisture of the 4 layers to 2.89 m (volumetric x depth), snow water equivalent, canopy water.
3. Groundwater storage anomaly GWSA = TWSA - (soil + snow + canopy). Surface water (rivers, reservoirs) and glacier
   mass are NOT yet removed: over the Indus and Ganges plains they are small; over the Third Pole glacier loss is the
   main signal and must be removed (Hugonnet et al. 2021) before any groundwater claim there (step 02).
4. Region series (cm and km3) and trends (Theil-Sen, cm/yr, with GRACE gap 2017-07 to 2018-05 left empty, not filled).
Writes data/processed/{twsa,gwsa}_0p5.nc, outputs/regions_monthly.csv, outputs/trends.csv, figures/*.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]
RAW, P, FIG, OUT = R / "data/raw", R / "data/processed", R / "figures", R / "outputs"
for d in (RAW, P, FIG, OUT):
    d.mkdir(parents=True, exist_ok=True)
BOX = (60.0, 5.0, 105.0, 40.0)          # west, south, east, north: South Asia + greater Third Pole
REGIONS = {                                           # simple boxes for a first look; basin polygons in step 03
    "Indus plains (Punjab, Sindh)": (67.0, 24.0, 75.5, 33.0),
    "NW India (Punjab, Haryana, Rajasthan, Delhi)": (72.5, 26.0, 78.5, 32.5),
    "Ganges plain (UP, Bihar)": (78.0, 24.0, 88.0, 30.0),
    "Bengal delta": (88.0, 21.5, 92.5, 26.0),
    "Upper Indus and Karakoram (mostly glacier)": (71.0, 34.0, 78.5, 37.5),
    "Central Tibetan Plateau": (82.0, 30.0, 95.0, 35.5),
    "Peninsular India (hard rock)": (74.0, 12.0, 82.0, 20.0),
}


def grace():
    have = sorted(RAW.glob("GRCTellus.JPL.*.nc"))
    if have:                                   # reuse the download; delete it to fetch a newer release
        f = have[-1]
    else:
        import earthaccess
        earthaccess.login(strategy="netrc")
        g = earthaccess.search_data(short_name="TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4", count=1)
        f = RAW / Path(g[0].data_links()[0]).name
        earthaccess.download(g, str(RAW))
    ds = xr.open_dataset(f)
    lon = ds.lon.values; ds = ds.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon")
    ds = ds.sel(lon=slice(BOX[0], BOX[2]), lat=slice(BOX[1], BOX[3]))
    tw = ds.lwe_thickness * ds.get("scale_factor", 1.0) if "scale_factor" in ds else ds.lwe_thickness
    tw = tw * ds.land_mask if "land_mask" in ds else tw
    tw.name = "twsa_cm"
    tw.to_netcdf(P / "twsa_0p5.nc")
    print("GRACE", f.name, tw.time.values[0], "->", tw.time.values[-1], tw.shape, flush=True)
    return tw


def era5_stores(times):
    f = P / "era5land_stores_anom_0p5.nc"
    if f.exists():
        return xr.open_dataset(f)
    import ee
    ee.Initialize(project="ee-ahmadammargeo")
    col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate("2002-01-01", "2027-01-01")
    def stores(i):
        sm = (i.select("volumetric_soil_water_layer_1").multiply(0.07).add(i.select("volumetric_soil_water_layer_2").multiply(0.21))
              .add(i.select("volumetric_soil_water_layer_3").multiply(0.72)).add(i.select("volumetric_soil_water_layer_4").multiply(1.89)))
        return ee.Image.cat(sm.multiply(100).rename("soil_cm"), i.select("snow_depth_water_equivalent").multiply(100).rename("snow_cm"),
                            i.select("skin_reservoir_content").multiply(100).rename("canopy_cm")).set("t", i.get("system:time_start"))
    col = col.map(stores)
    n = col.size().getInfo()
    lons = np.arange(BOX[0] + 0.25, BOX[2], 0.5); lats = np.arange(BOX[1] + 0.25, BOX[3], 0.5)
    arrs, ts = [], []
    imgs = col.toList(n)
    for k in range(n):
        img = ee.Image(imgs.get(k))
        req = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
               "grid": {"dimensions": {"width": len(lons), "height": len(lats)}, "crsCode": "EPSG:4326",
                        "affineTransform": {"scaleX": 0.5, "shearX": 0, "translateX": BOX[0], "shearY": 0, "scaleY": -0.5, "translateY": BOX[3]}}}
        a = ee.data.computePixels(req)
        arrs.append(np.stack([a["soil_cm"], a["snow_cm"], a["canopy_cm"]])[:, ::-1, :])   # south to north
        ts.append(pd.Timestamp(img.get("t").getInfo(), unit="ms"))
        if k % 24 == 0:
            print("ERA5-Land", ts[-1].date(), flush=True)
    a = np.array(arrs)
    ds = xr.Dataset({v: (("time", "lat", "lon"), a[:, i]) for i, v in enumerate(("soil_cm", "snow_cm", "canopy_cm"))},
                    coords={"time": pd.DatetimeIndex(ts) + pd.Timedelta(days=14), "lat": lats, "lon": lons})
    base = ds.sel(time=slice("2004-01-01", "2009-12-31")).mean("time")
    ds = ds - base
    ds.to_netcdf(P / "era5land_stores_anom_0p5.nc")
    return ds


def theil_sen(t, y):
    from scipy.stats import theilslopes
    ok = np.isfinite(y)
    if ok.sum() < 24:
        return np.nan, np.nan, np.nan
    s, i, lo, hi = theilslopes(y[ok], t[ok])
    return s, lo, hi


def main():
    tw = grace()
    st = era5_stores(tw.time)
    st = st.interp(lat=tw.lat, lon=tw.lon, method="nearest")
    tw = tw.assign_coords(time=tw.time.to_index().to_period("M").to_timestamp()).groupby("time").mean()   # two solutions in a few months: average
    tm = tw.time.to_index().to_period("M")
    stm = st.assign_coords(time=st.time.to_index().to_period("M"))
    stm = stm.sel(time=[t for t in tm if t in set(stm.time.to_index())])
    tw2 = tw.assign_coords(time=tm).sel(time=stm.time)
    gw = tw2 - (stm.soil_cm + stm.snow_cm + stm.canopy_cm)
    gw.name = "gwsa_cm"
    gw.assign_coords(time=gw.time.to_index().to_timestamp()).to_netcdf(P / "gwsa_0p5.nc")
    # region series
    area = np.cos(np.radians(gw.lat)) * (0.5 * 111.32) ** 2           # km2 per cell
    rows = []
    for name, (w, s, e, n) in REGIONS.items():
        sub = dict(lon=slice(w, e), lat=slice(s, n))
        a_ = area.sel(lat=slice(s, n))
        for var, da in (("twsa_cm", tw2), ("gwsa_cm", gw), ("soil_cm", stm.soil_cm), ("snow_cm", stm.snow_cm)):
            x = da.sel(**sub)
            m = (x * a_).sum(("lat", "lon"), skipna=True) / (a_ * x.notnull()).sum(("lat", "lon"))
            for t, v in zip(x.time.to_index(), m.values):
                rows.append({"region": name, "var": var, "month": str(t), "value": float(v)})
    df = pd.DataFrame(rows); df.to_csv(OUT / "regions_monthly.csv", index=False)
    tr = []
    for (reg, var), d in df.groupby(["region", "var"]):
        t = pd.PeriodIndex(d.month, freq="M").to_timestamp()
        yrs = (t - t[0]).days.values / 365.25
        s, lo, hi = theil_sen(yrs, d.value.values)
        tr.append({"region": reg, "var": var, "trend_cm_per_yr": round(s, 2), "lo": round(lo, 2), "hi": round(hi, 2),
                   "from": str(d.month.iloc[0]), "to": str(d.month.iloc[-1])})
    tr = pd.DataFrame(tr); tr.to_csv(OUT / "trends.csv", index=False)
    print(tr[tr["var"].isin(["twsa_cm", "gwsa_cm"])].to_string(index=False))
    figures(gw, df)


def figures(gw, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = gw.time.to_index().to_timestamp()
    yrs = (t - t[0]).days.values / 365.25
    a = gw.values.reshape(len(t), -1)
    slope = np.full(a.shape[1], np.nan)
    ok = np.isfinite(a).sum(0) > 60
    X = np.vstack([yrs, np.ones_like(yrs)]).T
    for j in np.flatnonzero(ok):
        m = np.isfinite(a[:, j]); slope[j] = np.linalg.lstsq(X[m], a[m, j], rcond=None)[0][0]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(gw.lon, gw.lat, slope.reshape(gw.shape[1:]), cmap="RdBu", vmin=-3, vmax=3, shading="auto")
    for name, (w, s, e, n) in REGIONS.items():
        ax.plot([w, e, e, w, w], [s, s, n, n, s], color="k", lw=0.6); ax.text(w + 0.2, n - 0.6, name.split(" (")[0], fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="trend (cm of water per year)")
    ax.set_title(f"Groundwater storage trend from GRACE/GRACE-FO minus ERA5-Land soil, snow, canopy, {t[0]:%Y-%m} to {t[-1]:%Y-%m}\n"
                 "(over the Third Pole this still includes glacier loss: not groundwater)", fontsize=10)
    fig.text(0.01, 0.01, "JPL mascon RL06.3 v04 CRI (0.5 deg grid, ~300 km true resolution); ERA5-Land monthly. Linear trend per cell.", fontsize=7, color="0.4")
    fig.savefig(FIG / "gwsa_trend_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(len(REGIONS), 1, figsize=(10, 2.1 * len(REGIONS)), sharex=True)
    for ax, (name, d) in zip(axs, df[df["var"].isin(["twsa_cm", "gwsa_cm"])].groupby("region", sort=False)):
        for var, dd in d.groupby("var"):
            ax.plot(pd.PeriodIndex(dd.month, freq="M").to_timestamp(), dd.value, lw=1, label={"twsa_cm": "total water", "gwsa_cm": "groundwater"}[var])
        ax.set_title(name, fontsize=9, loc="left"); ax.axhline(0, color="0.7", lw=0.5); ax.set_ylabel("cm")
    axs[0].legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(FIG / "regions_series.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
