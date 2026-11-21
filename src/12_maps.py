"""Publication maps of the groundwater results (step 12), one consistent style, written to figures/maps/.

    python src/12_maps.py          (after steps 01-11)

Style: equal-area projection (Lambert azimuthal, centred 82 E 23 N) so areas compare fairly; sea pale blue, land pale
grey, country outlines (from the GAUL admin-1 file, dissolved), a title, a one-line context under it, a legend with units,
and the data sources in a footer. Groundwater classes and colours are the same as in the QGIS project: brown = losing
water, blue = gaining, white = no clear change (-0.25 to +0.25 cm/yr). Units GRACE cannot resolve on its own are hatched.

Maps
  01 groundwater trend 2002-latest by unit         02 three periods side by side
  03 rainfall-driven part vs the rest, 2018-latest 04 CGWB well trends (points)
  05 wells by district                             06 wells vs GRACE by district
  07 GRACE's real resolution vs downscaled         08 kriged well surface and its uncertainty
  09 glacier mass change removed                   10 total water storage trend (what GRACE sees before removals)
  11 how much finer detail holds up (chart)
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]; Q = R / "outputs/qgis"; FO = R / "outputs/fine"; P = R / "data/processed"
OUT = R / "figures/maps"; OUT.mkdir(parents=True, exist_ok=True)
CRS = "+proj=laea +lon_0=82 +lat_0=23 +datum=WGS84 +units=m"
DIV = ["#8c2d04", "#cc4c02", "#ec7014", "#fec44f", "#f7f7f7", "#9ecae1", "#4292c6", "#08519c"]
BRK = [-99, -3, -2, -1, -0.25, 0.25, 1, 2, 99]
LAB = ["below -3 (fast loss)", "-3 to -2", "-2 to -1", "-1 to -0.25", "-0.25 to 0.25 (no clear change)", "0.25 to 1", "1 to 2", "above 2 (gain)"]
CMAP, NORM = ListedColormap(DIV), BoundaryNorm(BRK, len(DIV))
SEA, LAND, LINE = "#e6eef4", "#f2f0eb", "#8a8a8a"
SRC_G = "GRACE/GRACE-FO JPL RL06.3 CRI, CSR, GFZ mascons/grids minus ERA5-Land and GLDAS stores; glacier loss from Hugonnet et al. 2021"
SRC_W = "CGWB well levels (India Data Portal, ODC-By), May readings"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11})

A1 = gpd.read_file(P / "admin1.geojson").explode(index_parts=False)
A1 = A1[A1.geom_type == "Polygon"].to_crs(CRS)                 # dissolves can leave stray lines; keep the polygons
COUNTRIES = A1.dissolve("ADM0_NAME").reset_index()
NE = Path.home() / "dev/shared-geodata/data/natural_earth/ne_50m_land.geojson"
LANDMASS = gpd.read_file(NE).cx[40:120, -10:55].to_crs(CRS)      # all land, incl. Iran, Myanmar, Central Asia
VIEW = gpd.GeoSeries.from_xy([60, 100], [5, 38], crs=4326).to_crs(CRS).total_bounds


def base(ax, view=None):
    ax.set_facecolor(SEA)
    LANDMASS.plot(ax=ax, color=LAND, edgecolor="none")
    v = view if view is not None else VIEW
    ax.set_xlim(v[0], v[2]); ax.set_ylim(v[1], v[3]); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


def lines(ax):
    COUNTRIES.boundary.plot(ax=ax, color=LINE, lw=0.5)


def frame(fig, title, context, source, path):
    fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.935, context, ha="left", va="top", fontsize=9.5, color="#333333", wrap=True)
    fig.text(0.02, 0.012, "Data: " + source, ha="left", fontsize=7, color="#666666")
    fig.savefig(OUT / path, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", path)


def legend(ax, title="cm of water per year", loc="lower left", extra=()):
    h = [Patch(facecolor=c, edgecolor="#999999", label=l) for c, l in zip(DIV, LAB)] + list(extra)
    ax.legend(handles=h, title=title, loc=loc, fontsize=7, title_fontsize=8, frameon=True, framealpha=.9)


def legend_below(fig, title="cm of water per year", extra=()):
    """One legend under all panels, so no panel loses land to it."""
    h = [Patch(facecolor=c, edgecolor="#999999", label=l) for c, l in zip(DIV, LAB)] + list(extra)
    fig.legend(handles=h, title=title, loc="lower center", ncol=5, fontsize=8, title_fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.04))


def tight(fig, top=0.79, bottom=0.2):
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=bottom, wspace=0.03)


def units():
    u = gpd.read_file(Q / "groundwater.gpkg", layer="units").to_crs(CRS)
    u["geometry"] = u.buffer(3000).buffer(-3000)          # close slivers left by the admin-1 dissolve
    u["res"] = u.resolvable.astype(str) == "True"
    return u


def choro(ax, g, col, hatch_unres=True):
    g.plot(ax=ax, column=col, cmap=CMAP, norm=NORM, edgecolor="white", lw=0.6)
    if hatch_unres and "res" in g:
        g[~g.res].plot(ax=ax, facecolor="none", edgecolor="#333333", hatch="////", lw=0.6)


def raster(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype(float); b = s.bounds
    return a, (b.left, b.right, b.bottom, b.top)


def grid_plot(ax, a, ext, cmap, norm=None, vmin=None, vmax=None):
    """Plot a lon/lat grid in the map projection by projecting cell centres (pcolormesh)."""
    ny, nx = a.shape
    lon = np.linspace(ext[0], ext[1], nx + 1); lat = np.linspace(ext[3], ext[2], ny + 1)
    LON, LAT = np.meshgrid(lon, lat)
    from pyproj import Transformer
    X, Y = Transformer.from_crs(4326, CRS, always_xy=True).transform(LON, LAT)
    return ax.pcolormesh(X, Y, np.ma.masked_invalid(a), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, shading="flat", rasterized=True)


def m01():
    u = units(); fig, ax = plt.subplots(figsize=(10, 8.6)); base(ax); choro(ax, u, "trend_whole"); lines(ax)
    for _, r in u.iterrows():
        p = r.geometry.representative_point()
        ax.annotate(f"{r.trend_whole:+.1f}", (p.x, p.y), ha="center", fontsize=7.5, fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.7))
    legend(ax, extra=[Patch(facecolor="none", edgecolor="#333", hatch="////", label="too small for GRACE alone")])
    frame(fig, "Groundwater is being lost across the plains of north India and Pakistan",
          "Trend in groundwater storage, April 2002 to the latest GRACE-FO month, mean of an ensemble of GRACE solutions and land models,\n"
          "glacier loss removed. Numbers: cm of water per year spread over each region. Hatched: region too small for GRACE to resolve alone.",
          SRC_G, "01_groundwater_trend_units.png")


def m02():
    u = units(); fig, axs = plt.subplots(1, 3, figsize=(16, 6.2))
    for ax, (c, t) in zip(axs, [("trend_2002_2009", "2002-2009 (GRACE)"), ("trend_2010_2017", "2010-2017 (GRACE)"), ("trend_2018_latest", "2018-latest (GRACE-FO)")]):
        base(ax); choro(ax, u, c); lines(ax); ax.set_title(t, fontweight="bold")
    legend_below(fig, extra=[Patch(facecolor="none", edgecolor="#333", hatch="////", label="too small for GRACE alone")]); tight(fig)
    frame(fig, "Where the loss has eased, and where it has not",
          "Groundwater storage trend in three periods, cm of water per year (same classes in every panel). Loss in Indian Punjab and Haryana eased\n"
          "(-4.4, -2.3, -2.0) but has not stopped; Pakistan Punjab's loss picked up again after 2018 (-1.2); Gujarat and central India gained after 2018.",
          SRC_G, "02_groundwater_trend_periods.png")


def m03():
    u = units(); fig, axs = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, (c, t) in zip(axs, [("rain_2018", "Part explained by rainfall"), ("rest_2018", "The rest: mostly pumping")]):
        base(ax); choro(ax, u, c); lines(ax); ax.set_title(t, fontweight="bold")
    legend_below(fig, extra=[Patch(facecolor="none", edgecolor="#333", hatch="////", label="too small for GRACE alone")]); tight(fig)
    frame(fig, "Rain or people? Splitting the 2018-latest trend",
          "A rainfall model (each region's response to past rain, fitted to the GRACE record) gives the part of the trend that rain alone explains;\n"
          "the rest is what rain does not explain, chiefly pumping for irrigation. cm of water per year.",
          SRC_G + "; rainfall ERA5-Land", "03_rain_vs_rest_2018.png")


def m04():
    w = gpd.read_file(Q / "groundwater.gpkg", layer="wells").to_crs(CRS)
    br = [-99, -0.5, -0.1, 0.1, 0.5, 1, 99]; cols = ["#2166ac", "#92c5de", "#e0e0e0", "#f4a582", "#d6604d", "#67001f"]
    labs = ["rising > 0.5", "rising 0.1-0.5", "stable (+-0.1)", "falling 0.1-0.5", "falling 0.5-1", "falling > 1"]
    view = gpd.GeoSeries.from_xy([67, 98], [6, 36], crs=4326).to_crs(CRS).total_bounds
    fig, ax = plt.subplots(figsize=(10, 9.5)); base(ax, view); lines(ax)
    w = w.sort_values("slope", key=abs)
    w.plot(ax=ax, column="slope", cmap=ListedColormap(cols), norm=BoundaryNorm(br, len(cols)), markersize=3.5, lw=0)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, labs)], title="water table, m per year", loc="lower left", fontsize=7.5)
    frame(fig, f"{len(w):,} wells: the water table is falling fastest in Punjab, Haryana and Uttar Pradesh",
          "Trend of the May (pre-monsoon) depth to water, 2013-2020, per well with at least 5 years of readings. Red: water table falling.",
          SRC_W, "04_wells_points.png")


def districts():
    return gpd.read_file(Q / "fine.gpkg", layer="districts").to_crs(CRS)


def m05():
    d = districts(); view = gpd.GeoSeries.from_xy([67, 98], [6, 36], crs=4326).to_crs(CRS).total_bounds
    fig, ax = plt.subplots(figsize=(10, 9.5)); base(ax, view)
    d.boundary.plot(ax=ax, color="#d0d0d0", lw=0.15)
    d[d.wells.notna()].plot(ax=ax, column="storage_mid_cm_yr", cmap=CMAP, norm=NORM, edgecolor="white", lw=0.2); lines(ax)
    legend(ax, extra=[Patch(facecolor=LAND, edgecolor="#d0d0d0", label="fewer than 5 wells / no data")])
    frame(fig, "District by district: storage change measured in wells",
          "Median well trend in each district with 5 or more wells (383 districts), 2013-2020, times specific yield (alluvium 0.12, hard rock 0.02:\n"
          "the middle of each range). cm of water per year: the same units and classes as the GRACE maps.", SRC_W + "; districts FAO GAUL 2015",
          "05_wells_by_district.png")


def m06():
    d = districts(); view = gpd.GeoSeries.from_xy([67, 98], [6, 36], crs=4326).to_crs(CRS).total_bounds
    cats = {"both losing": "#8c2d04", "wells losing, GRACE not": "#ec7014", "GRACE losing, wells not": "#6a51a3", "neither losing": "#9ecae1"}
    fig, ax = plt.subplots(figsize=(10, 9.5)); base(ax, view); d.boundary.plot(ax=ax, color="#d0d0d0", lw=0.15)
    for k, c in cats.items():
        s = d[d.wells_vs_grace == k]
        if len(s):
            s.plot(ax=ax, color=c, edgecolor="white", lw=0.2)
    lines(ax); n = d.wells_vs_grace.value_counts()
    ax.legend(handles=[Patch(color=c, label=f"{k} ({n.get(k, 0)})") for k, c in cats.items()], title="districts, 2013-2020", loc="lower left", fontsize=8)
    frame(fig, "Wells and satellites mostly agree; where they do not, the loss is local",
          "Loss = below -0.25 cm of water per year. Orange: districts losing water by their wells while GRACE, which averages ~300 km, does not see it.\n"
          "Purple: GRACE loss from the wider region, with wells in the district steady.", SRC_W + "; " + "GRACE as map 01, same years", "06_wells_vs_grace_districts.png")


def m07():
    a, ext = raster(Q / "grace_trend_0p1_plain.tif"); b, _ = raster(Q / "gwsa_trend_0p1_downscaled.tif")
    view = gpd.GeoSeries.from_xy([68, 92], [21, 35], crs=4326).to_crs(CRS).total_bounds
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, arr, t in zip(axs, (a, b), ("What GRACE resolves: 3 degree blocks", "Downscaled to 0.1 degree: EXPERIMENTAL")):
        base(ax, view); grid_plot(ax, arr, ext, CMAP, NORM); lines(ax); ax.set_title(t, fontweight="bold")
    legend_below(fig); tight(fig)
    frame(fig, "Sharper is not the same as right",
          "Left: GRACE's groundwater trend 2002-latest on a 0.1 degree grid, showing the mascon blocks the mission actually solves for. Right: a random forest\n"
          "spreads each block's water by irrigation, heat, rain and population maps. Tested on wells, the extra detail does not hold up (map 11).",
          SRC_G + "; predictors MODIS, ERA5-Land, Copernicus land cover, GPWv4, GMTED", "07_grace_resolution_vs_downscaled.png")


def m08():
    a, ext = raster(Q / "wells_krige_slope.tif"); s, _ = raster(Q / "wells_krige_sd.tif")
    view = gpd.GeoSeries.from_xy([67, 98], [6, 36], crs=4326).to_crs(CRS).total_bounds
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.4))
    base(axs[0], view); m = grid_plot(axs[0], a, ext, "RdBu_r", vmin=-1, vmax=1); lines(axs[0]); axs[0].set_title("Smoothed water-table trend", fontweight="bold")
    fig.colorbar(m, ax=axs[0], shrink=.6, label="m per year (red = falling)")
    base(axs[1], view); m = grid_plot(axs[1], s, ext, "Greys", vmin=0.3, vmax=0.8); lines(axs[1]); axs[1].set_title("Its uncertainty", fontweight="bold")
    fig.colorbar(m, ax=axs[1], shrink=.6, label="standard deviation, m per year")
    frame(fig, "A surface from the wells: a tendency, not a measurement",
          "Ordinary kriging of well trends binned to 0.1 degree, drawn only within 30 km of wells. Cross-validation: it barely beats the average well\n"
          "(skill 0.01), because one well's trend is mostly local. Read it as the regional tendency, with the uncertainty beside it.",
          SRC_W, "08_wells_kriged.png")


def m09():
    a, ext = raster(Q / "glacier_cmyr.tif")
    fig, ax = plt.subplots(figsize=(10, 8.6)); base(ax); a[a == 0] = np.nan
    m = grid_plot(ax, a, ext, ListedColormap(["#3f007d", "#6a51a3", "#9e9ac8", "#dadaeb"]), BoundaryNorm([-30, -10, -3, -1, 0], 4)); lines(ax)
    fig.colorbar(m, ax=ax, shrink=.6, label="cm of water per year over each 0.5 degree cell")
    frame(fig, "Melting glaciers, taken out before counting groundwater",
          "Glacier mass change 2000-2019 spread over each 0.5 degree cell. GRACE weighs all water, so ice loss in the Karakoram, Himalaya and\n"
          "Nyainqentanglha would otherwise be read as groundwater loss.", "Hugonnet et al. 2021 glacier elevation change, x 850 kg per m3", "09_glacier_loss.png")


def m10():
    a, ext = raster(Q / "twsa_trend_jplcri.tif")
    fig, ax = plt.subplots(figsize=(10, 8.6)); base(ax); grid_plot(ax, a, ext, CMAP, NORM); lines(ax); legend(ax)
    frame(fig, "What GRACE weighs: all the water, before anything is removed",
          "Trend in total water storage (soil, snow, ice, rivers, lakes and groundwater) 2002-latest, JPL CRI mascons. Groundwater maps remove\n"
          "soil water, snow and canopy (land models) and glacier loss from this.", "GRACE/GRACE-FO JPL RL06.3 v04 CRI mascons", "10_total_water_storage.png")


def m11():
    s = pd.read_csv(FO / "downscale_skill.csv").dropna(subset=["level"])
    k = pd.read_csv(FO / "wells_krige_cv.csv")
    rows = [("Districts: GRACE alone", s.iloc[2].spearman_plain_grace), ("Districts: downscaled", s.iloc[2].spearman_downscaled),
            ("0.1 deg cells: GRACE alone", s.iloc[0].spearman_plain_grace), ("0.1 deg cells: downscaled", s.iloc[0].spearman_downscaled),
            ("Inside GRACE cells: downscaled", s.iloc[1].spearman_downscaled), ("Inside GRACE cells: trained on wells", s.iloc[3].spearman_downscaled),
            ("Kriging, random wells hidden", k.iloc[1].spearman), ("Kriging, 2 degree blocks hidden", k.iloc[0].spearman)]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    names, vals = zip(*rows)
    ax.barh(names[::-1], vals[::-1], color=["#9e9ac8" if "Kriging" in n else ("#4292c6" if "GRACE alone" in n else "#ec7014") for n in names[::-1]])
    for y, v in enumerate(vals[::-1]):
        ax.text(max(v, 0) + 0.01, y, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=.6); ax.set_xlim(-0.05, 0.6); ax.set_xlabel("rank correlation with wells held out (Spearman)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    frame(fig, "How much finer detail survives a test on wells?",
          "Blue: GRACE alone. Orange: downscaled. Purple: interpolated wells. District level works about as well with or without\n"
          "downscaling; detail inside GRACE's blocks does not match the wells.", SRC_W + "; outputs/fine/*.csv", "11_finer_detail_tests.png")


if __name__ == "__main__":
    for f in (m01, m02, m03, m04, m05, m06, m07, m08, m09, m10, m11):
        f()
