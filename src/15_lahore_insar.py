"""Land subsidence over Lahore from Sentinel-1 InSAR (step 15): a check on groundwater loss where well data are not public.

    python src/15_lahore_insar.py submit               # submit ASF HyP3 jobs (idempotent)
    python src/15_lahore_insar.py download           # when finished: download and unzip

Lahore pumps nearly all its water from the Indus-plain aquifer; a falling water table compacts the clay layers and the
ground sinks. Sinking ground is therefore a measurable, high-resolution (about 80 m) footprint of groundwater loss.
Same method as Karachi (commons-city src/karachi/15-16): INSAR_GAMMA 20x4 looks, every second acquisition paired with
its next two and with the one about a year later, then a MintPy small-baseline time series.
Stack: Sentinel-1 IW SLC, relative orbit 100 ascending, frame 97 (61 dates in 2023-2024 over the city).
Writes data/lahore/insar/hyp3/<job>/.
"""
import importlib.util
import sys
from pathlib import Path

import asf_search as asf

R = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("k", Path.home() / "dev/commons-city/src/karachi/15_insar_hyp3.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
k.NAME = "lahore_p100_f97_2023_2024"
k.OUT = R / "data/lahore/insar/hyp3"


def scenes():
    r = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1], processingLevel="SLC", beamMode="IW", relativeOrbit=100,
                       frame=97, intersectsWith="POINT(74.34 31.52)", start="2023-01-01", end="2025-01-01")
    one = {}
    for d, n in sorted({(p.properties["startTime"][:10], p.properties["sceneName"]) for p in r}):
        one.setdefault(d, n)
    return sorted(one.items())[::2]


k.scenes = scenes

if __name__ == "__main__":
    {"submit": k.submit, "download": k.download}[sys.argv[1]]()
