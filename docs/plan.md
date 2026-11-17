# Groundwater of South Asia and the greater Third Pole: past, present, downscaled, and scenarios

Box: 60-105 E, 5-40 N (Indus, Ganges-Brahmaputra-Meghna, peninsular India, Sri Lanka, Bangladesh, Nepal, Bhutan,
the Tibetan Plateau, Karakoram, Hindu Kush, Pamir edge). Monthly, 2002 to the latest GRACE-FO month, and back in
time where evidence allows.

## 1. What GRACE measures, and what it does not
- GRACE (2002-2017) and GRACE-FO (2018-) measure monthly changes in mass, so in total water storage (TWS): soil
  water + groundwater + snow + glaciers + rivers, lakes, reservoirs + canopy water.
- Mascon solutions (JPL RL06.3 v04 CRI, CSR RL06.2, GSFC) are gridded at 0.25-0.5 deg, but the real resolution is
  ~300 km. A grid cell is not a measurement at that cell.
- Gap: 2017-07 to 2018-05 between missions (left empty, never interpolated silently).

## 2. Isolating groundwater
GWSA = TWSA - (soil moisture + snow + canopy + surface water + glacier) anomalies.
- soil, snow, canopy: ERA5-Land monthly (to 2.89 m), with GLDAS Noah and CLSM as alternatives; the spread among
  them is part of the uncertainty.
- surface water: JRC Global Surface Water monthly extent x depth from DAHITI / Hydroweb / SWOT (2023-) altimetry for
  large reservoirs (Tarbela, Mangla, Bhakra, Tehri, Kaptai) and lakes.
- glaciers: Hugonnet et al. 2021 (2000-2019) and updates; regional mass loss (Karakoram near balance, Nyainqentanglha
  and eastern Himalaya strongly negative) removed by RGI region and spread over glacier area, then smoothed to the
  GRACE kernel. Without this, the Third Pole "groundwater" is glacier melt.
- GIA and leakage: GIA small here; leakage from ocean and from glaciers into the plains handled by the CRI mascons
  and tested with forward modelling.

## 3. Downscaling (from ~300 km to 1-10 km)
- Target: GWSA at 0.5 deg (monthly).
- Predictors at 1-10 km: precipitation (CHIRPS, IMERG), ET (MODIS MOD16, SSEBop, GLEAM), LST, NDVI and
  irrigated-area maps (MIRCA, GIAM, Landsat irrigation), canal command areas, crop calendars, soil and aquifer type
  (GLHYMPS permeability, BGS/IWMI aquifer maps), terrain, population and night lights (pumping), InSAR land
  subsidence (Sentinel-1 2015-, ERS/Envisat 1992-2010 where data exist), and the GRACE value of the parent cell.
- Models: random forest and gradient boosting (baselines), then a residual-correcting approach that conserves the
  GRACE mass in each parent cell (downscaling must not create or remove water).
- Validation, the part that matters: observation wells held out by region (spatially blocked):
  Punjab Irrigation Department piezometers (Pakistan), CGWB well levels (India, India-WRIS), BWDB (Bangladesh),
  and published well networks for Nepal Terai; converted to storage with specific yield ranges.

## 4. Looking back (before 2002)
- Climate-driven reconstructions of TWS (GRACE-REC, Humphrey and Gudmundsson 2019; Li et al. 2021) back to 1901 or
  1979: they capture climate variability but NOT pumping, which drives the Indus and Ganges declines.
- Pumping history from irrigated area (Landsat 1984-), tube-well counts (agricultural census), energy use, and
  subsidence from ERS-1/2 and Envisat. Wells are the backbone before 2002; models are checked against them.

## 5. Looking forward
- Scenarios, not forecasts: CMIP6 (SSP2-4.5, SSP3-7.0, SSP5-8.5) precipitation and temperature, crossed with
  water-use scenarios: current trend; canal lining and conjunctive use; crop shift away from rice and sugarcane in
  the arid Indus and NW India; and the solar-pump boom (cheap daytime pumping raises abstraction unless metered and
  capped: link to solar-wind-south-asia).
- Model: a lumped water-balance per aquifer unit calibrated on GRACE + wells, with an LSTM as a data-driven check.
- Report ranges, and the year each aquifer unit reaches a depth where pumping costs or water quality (arsenic,
  fluoride, salinity) make it unusable.

## 6. People
Who loses when water tables fall: smallholders who cannot deepen wells, women who fetch water, cities' poorest
neighbourhoods; and who gains from deeper wells. Groundwater use per district against land ownership and well
depth data (agricultural census) is the class question in this data.

## Steps
01 GRACE TWSA + ERA5-Land stores -> GWSA, region series and trends (done first pass)
02 glacier and surface water removal; GLDAS/CLSM alternatives; uncertainty
03 basins and aquifer units; well data assembly (Punjab, CGWB, BWDB)
04 downscaling and blocked validation
05 back-casting with reconstructions and wells
06 scenarios
07 maps for the Third Pole atlas (STAC, NetCDF), notebook and guide
