# Lessons

## CHIRPS rainfall has a step over Punjab in 2009-2010 (found 2026-11-21)
CHIRPS pentad totals at 75.5 E 30.8 N: about 375 mm a year in 2003-2008, about 800 in 2010-2024. ERA5-Land at the
same place shows no step (about 750 throughout, trend +46 mm per decade). The step is an artefact of the station data
CHIRPS blends in, not climate. Used as a predictor, the CHIRPS "rainfall trend" became the top variable in the
downscaling and had the wrong sign against wells. Fix: ERA5-Land rainfall in step 08; results re-run.
Rule: check any trend from a gridded product against a second, independent source before using it.

## Sharper is not right
Downscaled maps look detailed; tested on held-out wells, the detail inside GRACE cells does not hold up
(rank correlation about 0.02). Test the added detail itself (anomalies within the coarse cell), not only the totals.

## Single-well trends are local
Kriging of well trends barely beats the mean in cross-validation: aggregate (district medians), do not interpolate.
