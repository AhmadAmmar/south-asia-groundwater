"""Fine-scale inputs (step 08): predictors on a 0.1 deg grid and district boundaries, from Earth Engine.

    python src/08_fine_inputs.py

Grid: 60-105 E, 5-40 N at 0.1 deg (450 x 350 cells, ~11 km), cell edges on whole tenths, so 5 x 5 fine cells make one
0.5 deg GRACE cell exactly. Each predictor is the mean of the source pixels inside a cell (Earth Engine mean pyramids; all
predictors are continuous, so these are true area means), over 2003-2024 unless stated.

Predictors (why each one can tell where groundwater is being lost inside a GRACE cell)
  ndvi_dry        MODIS MOD13A2 NDVI, Jan-Apr mean: green crops in the dry season are almost always irrigated, and in
                  the Indus and Ganges plains most irrigation water is pumped groundwater
  ndvi_dry_trend  change in that (per decade): expanding or intensifying dry-season irrigation
  cropland        Copernicus Global Land Cover 100 m (2019), cropland cover fraction (0-1)
  et              MODIS MOD16A2GF evapotranspiration, mm/yr: water leaving the land, including pumped irrigation water
  precip          CHIRPS rainfall, mm/yr: recharge supply
  precip_trend    CHIRPS rainfall trend, mm/yr per decade
  lst_dry         MODIS MOD11A2 daytime land surface temperature, Mar-May, C: hot dry land vs cooled irrigated land
  pop             GPWv4.11 population density 2020, log10(people/km2 + 1): domestic and urban pumping
  elev, slope     SRTM / GMTED elevation (m) and slope (degrees): plains with aquifers vs mountains
Writes data/processed/fine_predictors_0p1.nc and data/processed/districts.geojson (FAO GAUL 2015 level 2, simplified
to ~500 m; GAUL boundaries are the UN's, used for aggregation only).
"""
import io
import sys
from pathlib import Path

import numpy as np
import requests
import xarray as xr

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]; P = R / "data/processed"
BOX = (60.0, 5.0, 105.0, 40.0); RES = 0.1
NX, NY = int(round((BOX[2] - BOX[0]) / RES)), int(round((BOX[3] - BOX[1]) / RES))
LONS = BOX[0] + RES / 2 + RES * np.arange(NX); LATS = BOX[3] - RES / 2 - RES * np.arange(NY)   # north to south
COUNTRIES = ["India", "Pakistan", "Bangladesh", "Nepal", "Bhutan", "Sri Lanka", "Afghanistan", "China", "Myanmar"]


def ee_init():
    import ee
    ee.Initialize(project="ee-ahmadammargeo")
    return ee


def predictors(ee):
    yrs = ee.Filter.calendarRange(2003, 2024, "year")
    def season(col, band, m0, m1, scale=1.0, offset=0.0):
        return ee.ImageCollection(col).filter(yrs).filter(ee.Filter.calendarRange(m0, m1, "month")).select(band).mean().multiply(scale).add(offset)
    def yearly_trend(col, band, m0, m1, scale, reducer="mean"):
        imgs = ee.List.sequence(2003, 2024).map(lambda y: ee.ImageCollection(col).filter(ee.Filter.calendarRange(y, y, "year"))
                                                 .filter(ee.Filter.calendarRange(m0, m1, "month")).select(band)
                                                 .reduce(reducer).multiply(scale).rename("v")
                                                 .addBands(ee.Image.constant(ee.Number(y).subtract(2003)).float().rename("t")))
        return ee.ImageCollection(imgs).select(["t", "v"]).reduce(ee.Reducer.linearFit()).select("scale").multiply(10)
    ndvi_dry = season("MODIS/061/MOD13A2", "NDVI", 1, 4, 1e-4)
    ndvi_tr = yearly_trend("MODIS/061/MOD13A2", "NDVI", 1, 4, 1e-4)
    crop = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019").select("crops-coverfraction").divide(100)
    et = ee.ImageCollection("MODIS/061/MOD16A2GF").filter(yrs).select("ET").mean().multiply(0.1 * 365 / 8)   # kg/m2/8d -> mm/yr
    chirps = "UCSB-CHG/CHIRPS/PENTAD"
    pr = ee.ImageCollection(chirps).filter(yrs).select("precipitation").sum().divide(22)
    pr_tr = yearly_trend(chirps, "precipitation", 1, 12, 1, "sum")
    lst = season("MODIS/061/MOD11A2", "LST_Day_1km", 3, 5, 0.02, -273.15)
    pop = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density").filter(ee.Filter.eq("system:index", "gpw_v4_population_density_rev11_2020_30_sec")).first().add(1).log10()
    dem = ee.Image("USGS/GMTED2010_FULL").select("min")
    slope = ee.Terrain.slope(dem)
    return {"ndvi_dry": ndvi_dry, "ndvi_dry_trend": ndvi_tr, "cropland": crop, "et": et, "precip": pr, "precip_trend": pr_tr,
            "lst_dry": lst, "pop": pop, "elev": dem, "slope": slope}


def fetch(ee, img, name):
    """One band on the 0.1 deg grid (area-weighted mean of source pixels) as a numpy array, north up."""
    import rasterio
    img = img.rename("b").toFloat().unmask(-9999)
    url = img.getDownloadURL({"crs": "EPSG:4326", "crs_transform": [RES, 0, BOX[0], 0, -RES, BOX[3]],
                              "dimensions": f"{NX}x{NY}", "format": "GEO_TIFF"})
    r = requests.get(url, timeout=900); r.raise_for_status()
    with rasterio.open(io.BytesIO(r.content)) as s:
        a = s.read(1).astype("float32")
    a[a <= -9998] = np.nan
    print(f"{name:15s} min {np.nanmin(a):9.3f}  median {np.nanmedian(a):9.3f}  max {np.nanmax(a):9.3f}", flush=True)
    return a


def districts(ee):
    import geopandas as gpd
    fc = (ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.inList("ADM0_NAME", COUNTRIES))
          .filterBounds(ee.Geometry.Rectangle(list(BOX)))
          .map(lambda f: f.simplify(500).select(["ADM0_NAME", "ADM1_NAME", "ADM2_NAME", "ADM2_CODE"])))
    url = fc.getDownloadURL(filetype="geojson")
    g = gpd.read_file(io.BytesIO(requests.get(url, timeout=900).content))
    g = g[g.is_valid | ~g.is_valid].copy(); g["geometry"] = g.geometry.buffer(0)
    g.to_file(P / "districts.geojson", driver="GeoJSON")
    print("districts", len(g), "in", g.ADM0_NAME.nunique(), "countries")


def main():
    ee = ee_init()
    f = P / "fine_predictors_0p1.nc"
    if not f.exists():
        arrs = {k: fetch(ee, img, k) for k, img in predictors(ee).items()}
        ds = xr.Dataset({k: (("lat", "lon"), v) for k, v in arrs.items()}, coords={"lat": LATS, "lon": LONS})
        ds.attrs["note"] = "0.1 deg means of source pixels, 2003-2024; see src/08_fine_inputs.py"
        ds.to_netcdf(f)
    if not (P / "districts.geojson").exists():
        districts(ee)


if __name__ == "__main__":
    main()
