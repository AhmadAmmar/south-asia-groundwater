"""Inputs for the uncertainty-aware groundwater estimate (step 02): alternative GRACE solutions, alternative land
models for the non-groundwater stores, glacier mass change, and observation-well trends.

    python src/02_inputs.py grace     # CSR and GFZ land grids (GRACE + GRACE-FO, PO.DAAC) and JPL mascon without CRI (EE)
    python src/02_inputs.py stores    # GLDAS-2.1 Noah and GLDAS-2.2 CLSM soil + snow + canopy, monthly, 0.5 deg (EE)
    python src/02_inputs.py glacier   # Hugonnet et al. 2021 glacier elevation change 2000-2019 -> cm w.e./yr per 0.5 deg cell
    python src/02_inputs.py wells     # Jasechko et al. 2024 well-level data (Zenodo 10003697)

GRACE
  JPL RL06.3 v04 CRI mascons (step 01, with land scaling factors)                    -> data/processed/twsa_0p5.nc
  JPL RL06.3 v04 mascons without CRI (EE NASA/GRACE/MASS_GRIDS_V04/MASCON, to 2024-09) -> twsa_jplnocri_0p5.nc
  CSR and GFZ RL06 land grids, spherical harmonics, 1 deg (PO.DAAC TELLUS_GRAC_L3_*_RL06_LND_v04 and
  TELLUS_GRFO_L3_*_RL06.3_LND_v04), with their gain (scale) factors applied                -> twsa_csr_1deg.nc, twsa_gfz_1deg.nc
Stores (anomalies about 2004-2009, cm)
  GLDAS-2.1 Noah 3-hourly -> monthly: soil moisture 0-200 cm (4 layers), SWE, canopy  -> stores_noah_0p5.nc
  GLDAS-2.2 CLSM daily -> monthly: root-zone + surface soil profile (SoilMoist_P), SWE, canopy (its groundwater store is
  NOT used: CLSM assimilates GRACE)                                                     -> stores_clsm_0p5.nc
Glacier: 20-year mean dh/dt (m/yr) on glacier pixels at 100 m, averaged over each 0.5 deg cell with non-glacier = 0,
  x 850 kg m-3 (Huss 2013 volume-to-mass) -> cm water per year spread over the cell     -> glacier_cmyr_0p5.nc
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]
RAW, P = R / "data/raw", R / "data/processed"
BOX = (60.0, 5.0, 105.0, 40.0)
LONS = np.arange(BOX[0] + 0.25, BOX[2], 0.5); LATS = np.arange(BOX[1] + 0.25, BOX[3], 0.5)


def _ee():
    import ee
    ee.Initialize(project="ee-ahmadammargeo")
    return ee


def _pixels(ee, img, bands, scale=0.5):
    """One multi-band ee.Image on the 0.5 deg box grid as a dict of south-to-north arrays."""
    lons = np.arange(BOX[0] + scale / 2, BOX[2], scale); lats = np.arange(BOX[1] + scale / 2, BOX[3], scale)
    req = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
           "grid": {"dimensions": {"width": len(lons), "height": len(lats)}, "crsCode": "EPSG:4326",
                    "affineTransform": {"scaleX": scale, "shearX": 0, "translateX": BOX[0], "shearY": 0, "scaleY": -scale, "translateY": BOX[3]}}}
    for k in range(6):
        try:
            a = ee.data.computePixels(req); break
        except Exception as e:                  # transient EE errors: wait and retry
            import time; print("  retry", str(e)[:80], flush=True); time.sleep(20 * (k + 1))
    return {b: np.array(a[b], dtype="float32")[::-1, :] for b in bands}


def _monthly(ee, coll, start, end, fn, bands, out, label):
    """Monthly means of fn(image) from a sub-monthly collection, fetched month by month (resumable), as anomalies."""
    part = P / (out + ".partial.npz")
    done = dict(np.load(part, allow_pickle=True)) if part.exists() else {}
    months = pd.period_range(start, end, freq="M")
    for m in months:
        k = str(m)
        if k in done:
            continue
        c = ee.ImageCollection(coll).filterDate(str(m.start_time.date()), str((m + 1).start_time.date()))
        img = c.map(fn).mean()
        a = _pixels(ee, img, bands)
        done[k] = np.stack([a[b] for b in bands])
        if m.month == 1:
            print(f"  {label} {k}", flush=True); np.savez_compressed(part, **done)
    np.savez_compressed(part, **done)
    keys = [str(m) for m in months]
    a = np.array([done[k] for k in keys])
    ds = xr.Dataset({b: (("time", "lat", "lon"), a[:, i]) for i, b in enumerate(bands)},
                    coords={"time": pd.PeriodIndex(keys, freq="M").to_timestamp(), "lat": LATS, "lon": LONS})
    ds = ds - ds.sel(time=slice("2004-01-01", "2009-12-31")).mean("time")
    ds.to_netcdf(P / (out + ".nc")); print("wrote", out, flush=True)


def stores():
    """GLDAS Noah 2.1 and CLSM 2.2 non-groundwater stores, monthly anomalies in cm."""
    ee = _ee()
    def noah(i):
        sm = (i.select("SoilMoi0_10cm_inst").add(i.select("SoilMoi10_40cm_inst")).add(i.select("SoilMoi40_100cm_inst"))
              .add(i.select("SoilMoi100_200cm_inst")))                       # kg m-2 = mm
        return ee.Image.cat(sm.divide(10).rename("soil_cm"), i.select("SWE_inst").divide(10).rename("snow_cm"),
                            i.select("CanopInt_inst").divide(10).rename("canopy_cm"))
    def clsm(i):
        return ee.Image.cat(i.select("SoilMoist_P_tavg").divide(10).rename("soil_cm"), i.select("SWE_tavg").divide(10).rename("snow_cm"),
                            i.select("CanopInt_tavg").divide(10).rename("canopy_cm"))
    b = ["soil_cm", "snow_cm", "canopy_cm"]
    if not (P / "stores_clsm_0p5.nc").exists():
        _monthly(ee, "NASA/GLDAS/V022/CLSM/G025/DA1D", "2003-02", "2026-07", clsm, b, "stores_clsm_0p5", "CLSM")
    if not (P / "stores_noah_0p5.nc").exists():
        _monthly(ee, "NASA/GLDAS/V021/NOAH/G025/T3H", "2002-04", "2026-09", noah, b, "stores_noah_0p5", "Noah")


def grace():
    """JPL mascon without CRI from EE; CSR and GFZ land grids (GRACE and GRACE-FO) from PO.DAAC."""
    f = P / "twsa_jplnocri_0p5.nc"
    if not f.exists():
        ee = _ee()
        c = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON")
        ts = c.aggregate_array("system:time_start").getInfo()
        lst = c.toList(len(ts)); arr = []
        for k, t in enumerate(ts):
            arr.append(_pixels(ee, ee.Image(lst.get(k)).select("lwe_thickness"), ["lwe_thickness"])["lwe_thickness"])
        ds = xr.DataArray(np.array(arr), dims=("time", "lat", "lon"), name="twsa_cm",
                          coords={"time": pd.to_datetime(ts, unit="ms").to_period("M").to_timestamp(), "lat": LATS, "lon": LONS})
        ds.groupby("time").mean().to_netcdf(f); print("wrote", f.name, flush=True)
    import earthaccess
    earthaccess.login(strategy="netrc")
    for centre in ("CSR", "GFZ"):
        out = P / f"twsa_{centre.lower()}_1deg.nc"
        if out.exists():
            continue
        d = RAW / f"tellus_{centre.lower()}"; d.mkdir(exist_ok=True)
        for sn in (f"TELLUS_GRAC_L3_{centre}_RL06_LND_v04", f"TELLUS_GRFO_L3_{centre}_RL06.3_LND_v04"):
            g = earthaccess.search_data(short_name=sn)
            print(sn, len(g), "granules", flush=True)
            earthaccess.download(g, str(d))
        fs = sorted(d.glob("*.nc")); das = []
        for x in fs:
            ds = xr.open_dataset(x)
            lon = ds.lon.values; ds = ds.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon")
            das.append(ds.lwe_thickness.sel(lon=slice(*BOX[::2]), lat=slice(BOX[1], BOX[3])).load())
        da = xr.concat(das, "time")
        da = da.assign_coords(time=da.time.to_index().to_period("M").to_timestamp()).groupby("time").mean()
        sf = sorted(RAW.glob(f"tellus_{centre.lower()}/*SCALE*")) or sorted(RAW.glob("**/CLM4.SCALE_FACTOR*.nc"))
        if sf:
            s = xr.open_dataset(sf[0]); lon = s.lon.values
            s = s.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon")
            da = da * s["scale_factor"].sel(lon=slice(*BOX[::2]), lat=slice(BOX[1], BOX[3])).reindex_like(da, method="nearest")
        else:
            print("  WARNING: no gain factor file found for", centre, "- unscaled (signal will be damped)", flush=True)
        da = da - da.sel(time=slice("2004-01-01", "2009-12-31")).mean("time")
        da.name = "twsa_cm"; da.to_netcdf(out); print("wrote", out.name, da.shape, flush=True)


def glacier():
    """Hugonnet 2000-2019 glacier thinning as cm w.e./yr averaged over each 0.5 deg cell."""
    f = P / "glacier_cmyr_0p5.nc"
    if f.exists():
        return
    ee = _ee()
    c = (ee.ImageCollection("projects/sat-io/open-datasets/GLOBAL-GLACIER-MASS-LOSS/elevation-change")
         .filterBounds(ee.Geometry.Rectangle(BOX)).filter(ee.Filter.stringContains("system:index", "2000-01-01_2020-01-01")))
    n = c.size().getInfo(); print("tiles", n, flush=True)
    def per_tile(im):
        """Mean dh/dt (non-glacier = 0) over the four 0.5 deg cells of one 1 x 1 deg tile."""
        b = ee.List(im.geometry().bounds().coordinates().get(0))
        w = ee.Number(ee.List(b.get(0)).get(0)).round(); s = ee.Number(ee.List(b.get(0)).get(1)).round()
        cells = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([w.add(i * .5), s.add(j * .5), w.add(i * .5 + .5), s.add(j * .5 + .5)], None, False),
                                                 {"lon": w.add(i * .5 + .25), "lat": s.add(j * .5 + .25)}) for i in (0, 1) for j in (0, 1)])
        return im.select(0).unmask(0).rename("dh").reduceRegions(cells, ee.Reducer.mean(), 100, tileScale=4)
    lst = c.toList(n); rows = []
    import time
    for k in range(0, n, 4):                    # small batches: EE limits concurrent aggregations
        fc = ee.FeatureCollection(ee.List.sequence(k, min(k + 3, n - 1)).map(lambda i: per_tile(ee.Image(lst.get(i))))).flatten()
        for tries in range(8):
            try:
                rows += [f["properties"] for f in fc.getInfo()["features"]]; break
            except Exception as e:
                print("  retry", str(e)[:60], flush=True); time.sleep(15 * (tries + 1))
        if k % 40 == 0:
            print("  tiles", k, flush=True)
    a = np.zeros((len(LATS), len(LONS)), "float32")
    for r in rows:
        i, j = int((r["lat"] - BOX[1]) / 0.5), int((r["lon"] - BOX[0]) / 0.5)
        if 0 <= i < len(LATS) and 0 <= j < len(LONS) and r.get("mean") is not None:
            a[i, j] += r["mean"] * 0.85 * 100                                    # m/yr ice -> cm/yr water
    xr.DataArray(a, dims=("lat", "lon"), coords={"lat": LATS, "lon": LONS}, name="glacier_cm_per_yr").to_netcdf(f)
    print("glacier cm/yr: min %.2f, cells with loss %d" % (np.nanmin(a), int((a < -0.01).sum())), flush=True)


def precip():
    """ERA5-Land monthly precipitation (mm/month) on the 0.5 deg grid, 2000-01 to the latest month, for step 04."""
    ee = _ee()
    f = P / "precip_era5land_0p5.nc"
    if f.exists():
        return
    col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate("2000-01-01", "2027-01-01").select("total_precipitation_sum")
    ts = col.aggregate_array("system:time_start").getInfo(); lst = col.toList(len(ts)); arr = []
    for k, t in enumerate(ts):
        arr.append(_pixels(ee, ee.Image(lst.get(k)).multiply(1000).rename("p"), ["p"])["p"])
        if k % 24 == 0:
            print("  precip", pd.Timestamp(t, unit="ms").date(), flush=True)
    xr.DataArray(np.array(arr), dims=("time", "lat", "lon"), name="precip_mm",
                 coords={"time": pd.to_datetime(ts, unit="ms").to_period("M").to_timestamp(), "lat": LATS, "lon": LONS}).to_netcdf(f)
    print("wrote", f.name, flush=True)


def wells():
    """Download and unpack the Jasechko et al. 2024 groundwater-level and aquifer files."""
    import urllib.request
    d = RAW / "jasechko2024"; d.mkdir(exist_ok=True)
    for n in ("GroundwaterLevelData", "AquiferSpatialData", "SupplementaryTables"):
        z = d / f"Jasechko_et_al_2024_{n}.zip"
        if not z.exists():
            urllib.request.urlretrieve(f"https://zenodo.org/api/records/10003697/files/{z.name}/content", z)
        zipfile.ZipFile(z).extractall(d / n)
    for p in sorted(d.rglob("*"))[:40]:
        if p.is_file():
            print(p.relative_to(d), round(p.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    {"grace": grace, "stores": stores, "glacier": glacier, "wells": wells, "precip": precip}[sys.argv[1]]()
