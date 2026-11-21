"""QGIS project of the groundwater results, with plain names, units and diverging colour ramps centred on zero.

    "C:/Program Files/QGIS 4.2.3/bin/python-qgis.bat" src/07_qgis_project.py
Needs src/06_qgis_prep.py first. Uses the styling helpers of commons-city (engines/qgiskit.py).
Writes outputs/qgis/groundwater.qgz.

Groups: units by period (2002-09, 2010-17, 2018-latest, whole record; ensemble mean, cm/yr), rain and the rest
(step 04), wells (CGWB May depth trend 2013-2020), grids (one member's per-cell trend, total water storage, glacier
loss), basemap. Units that GRACE cannot resolve on their own are hatched.
"""
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "dev/commons-city/src"))
from engines import qgiskit as qk  # noqa: E402

O = R / "outputs/qgis"; G = str(O / "groundwater.gpkg")
DIV = ["#8c2d04", "#cc4c02", "#ec7014", "#fec44f", "#f7f7f7", "#9ecae1", "#4292c6", "#08519c"]
BRK = [-99, -3, -2, -1, -0.25, 0.25, 1, 2, 99]
LAB = ["below -3 (fast loss)", "-3 to -2", "-2 to -1", "-1 to -0.25", "-0.25 to 0.25 (no clear change)", "0.25 to 1", "1 to 2", "above 2 (gain)"]

P, root = qk.start("EPSG:4326", "Groundwater storage change, South Asia and the Third Pole")
groups = {n: root.addGroup(n) for n in ("Units: groundwater trend (cm of water per year)", "Rain or people (step 04)",
                                        "Wells (CGWB)", "Grids", "Basemap")}
for field, name, vis in (("trend_whole", "Whole record 2002-2026 (ensemble mean)", True), ("trend_2002_2009", "2002-2009", False),
                         ("trend_2010_2017", "2010-2017", False), ("trend_2018_latest", "2018-latest (GRACE-FO)", False)):
    l = qk.vec(G, "units", name); qk.grad(l, field, BRK, DIV, LAB, opacity=.85, outline=True); qk.label(l, "unit", 7)
    qk.add(P, l, groups["Units: groundwater trend (cm of water per year)"], visible=vis)
nr = qk.vec(G, "units", "Too small for GRACE alone (gain < 0.5 or > half the signal from neighbours)", subset="resolvable = 'False'")
qk.single(nr, None, "#000000", 0.8); nr.renderer().symbol().symbolLayer(0).setStrokeStyle(qk.Qt.PenStyle.DashLine)
qk.add(P, nr, groups["Units: groundwater trend (cm of water per year)"])
for field, name in (("rest_2018", "Not explained by rain, 2018-latest"), ("rest_2002", "Not explained by rain, 2002-2009"),
                    ("rain_2018", "Rainfall-driven part, 2018-latest")):
    l = qk.vec(G, "units", name); qk.grad(l, field, BRK, DIV, LAB, opacity=.85, outline=True)
    qk.add(P, l, groups["Rain or people (step 04)"], visible=False)
w = qk.vec(G, "wells", "CGWB wells: May depth trend 2013-2020 (m/yr; red = water table falling)")
qk.grad(w, "slope", [-99, -0.5, -0.1, 0.1, 0.5, 1, 99], ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#d6604d", "#67001f"],
        ["rising > 0.5", "rising 0.1-0.5", "stable", "falling 0.1-0.5", "falling 0.5-1", "falling > 1"], opacity=1, outline=False)
w.renderer().setSymbolSizes(1.4, 1.4)
qk.add(P, w, groups["Wells (CGWB)"], visible=False)
for f, name, st in (("gwsa_trend_jplcri_era5.tif", "Groundwater trend per cell, one member: JPL CRI - ERA5-Land (cm/yr; pattern only)",
                     [(-4, "#8c2d04", "-4"), (-2, "#ec7014", "-2"), (0, "#f7f7f7", "0"), (2, "#4292c6", "+2"), (4, "#08519c", "+4")]),
                    ("twsa_trend_jplcri.tif", "Total water storage trend, GRACE JPL CRI (cm/yr)",
                     [(-4, "#8c2d04", "-4"), (-2, "#ec7014", "-2"), (0, "#f7f7f7", "0"), (2, "#4292c6", "+2"), (4, "#08519c", "+4")]),
                    ("glacier_cmyr.tif", "Glacier mass change 2000-2019, Hugonnet et al. 2021 (cm water/yr over the cell)",
                     [(-10, "#3f007d", "-10"), (-3, "#807dba", "-3"), (-0.5, "#dadaeb", "-0.5"), (0, "#ffffff", "0")])):
    l = qk.ras(str(O / f), name); qk.pseudo(l, 1, st, opacity=.8); qk.add(P, l, groups["Grids"], visible=False)
qk.add(P, qk.xyz(qk.CARTO_LABELS, "Place labels (CARTO)"), groups["Basemap"])
qk.add(P, qk.xyz(qk.ESRI, "Satellite imagery (Esri)"), groups["Basemap"], visible=False)
qk.set_view(P, qk.rect_lonlat([60, 5, 105, 40], "EPSG:4326"), "EPSG:4326")
P.write(str(O / "groundwater.qgz")); print("wrote", O / "groundwater.qgz")
qk.stop()
