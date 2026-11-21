# Step 02-03 results: groundwater storage change by unit, with honest uncertainty

Run: `python src/02_inputs.py {grace,stores,glacier,wells}` then `python src/03_ensemble_units.py`.
Tables: `outputs/units_trends.csv` (ensemble), `units_members.csv` (every member), `units_leakage.csv`, `units_monthly.csv`.
Figures: `figures/units_trends.png`, `figures/units_series.png`.

## What changed from step 01

| Step 01 | Now | Why it matters |
|---|---|---|
| One GRACE solution (JPL mascon CRI) | Four: JPL CRI, JPL, CSR, GFZ | Processing choices change trends by up to ~1 cm/yr in small units |
| One land model (ERA5-Land) | ERA5-Land and GLDAS Noah; GLDAS CLSM as a sensitivity only | CLSM assimilates GRACE, so its soil already contains part of the groundwater loss; subtracting it removes real depletion (it halves NW India's trend) |
| Glacier mass left in | Hugonnet et al. (2021) 2000-2019 thinning removed (-17 Gt/yr in the box) | Mountain "groundwater" was mostly ice |
| Overlapping boxes | 20 non-overlapping units of admin-1 regions | No double counting; units match where water is governed |
| One straight line, 2002-2026 | Trends for 2002-09, 2010-17 (GRACE) and 2018-latest (GRACE-FO), plus the whole record | The decline is not steady |
| Theil-Sen interval (+/-0.1 cm/yr) | Autocorrelation-adjusted SE combined with the spread between members | Real uncertainty is +/-0.3 to 1.6 cm/yr |
| No resolution check | Leakage test per unit (gain, share of signal from neighbours) | Units under GRACE's ~300 km resolution are flagged |

CSR and GFZ land grids arrive in metres and without gain factors (the gain-factor file is no longer listed on PO.DAAC);
they were converted to cm and given gains computed from GLDAS Noah as in Landerer and Swenson (2012). The CSR RL06.3
mascons could not be downloaded (the server's certificate was rejected) and are not in the ensemble.

## Main results (cm of water per year, 95% interval; km3/yr over the unit)

| Unit | 2002-09 | 2010-17 | 2018-latest | Whole record | km3/yr |
|---|---|---|---|---|---|
| Indian Punjab, Haryana, Delhi (not resolvable alone) | -4.4 | -2.3 | -2.0 | -3.4 +/- 1.6 | -3.1 |
| Uttar Pradesh, Uttarakhand | -2.9 | -1.8 | -1.1 | -2.1 +/- 0.6 | -6.2 |
| Rajasthan | -1.9 | -1.2 | -0.1 | -1.5 +/- 1.0 | -5.3 |
| Pakistan Punjab | -1.3 | -0.5 | -1.2 | -0.9 +/- 0.5 | -2.0 |
| Bihar, Jharkhand | -1.3 | -0.6 | +0.3 | -1.0 +/- 0.7 | -1.7 |
| Bengal basin | -0.9 | -0.4 | +0.1 | -0.7 +/- 0.4 | -1.5 |
| Afghanistan | -0.8 | -0.6 | -0.8 | -0.6 +/- 0.3 | -4.2 |
| Northeast India, Bhutan, Sikkim | -1.2 | -0.7 | -0.9 | -1.0 +/- 0.4 | -3.1 |
| Maharashtra | +0.5 | +0.5 | +1.3 | +0.5 +/- 0.5 | +1.5 |
| Gujarat | +0.6 | +0.4 | +2.0 | +0.6 +/- 0.9 | +1.0 |
| South India (hard rock) | +0.3 | -0.7 | +1.7 | +0.1 +/- 1.1 | +0.3 |
| Western Himalaya, Karakoram (after glacier removal) | -0.1 | -0.1 | -1.2 | -0.1 +/- 0.7 | -0.3 |

Northwest India (Punjab, Haryana, Delhi and Rajasthan): -10.7 +/- 5.4 km3/yr in 2002-09, -6.4 in 2010-17 and -2.1 in
2018-latest (intervals widen because GRACE-FO is shorter). The Indus-Ganges plains as a whole (both Punjabs, Haryana,
Delhi, Rajasthan, UP, Bihar): about -24, -14 and -7 km3/yr over the three periods.

## Checks against published work

- Rodell et al. (2009), Rajasthan + Punjab + Haryana + Delhi, Aug 2002-Oct 2008: -17.7 +/- 4.5 km3/yr. Ours for
  2002-09: -10.7 +/- 5.4. The intervals overlap. Ours is lower partly because a unit mean does not add back signal
  that leaks out of a unit smaller than GRACE's footprint (gain 0.42 for Punjab-Haryana-Delhi, 0.66 for Rajasthan);
  Rodell et al. averaged over a larger kernel and scaled for this.
- Asoka et al. (2017): storage rising in parts of central and south India after 2002 with stronger monsoons and
  managed recharge. We find the same in Maharashtra, Gujarat and Madhya Pradesh, strongest since 2018.
- Well data: open annual well records for India, Pakistan and Bangladesh are not in the public Jasechko et al. (2024)
  release (only 71 wells near 60-70 E). Validation against wells still needs CGWB / India-WRIS, Punjab Irrigation
  Department and BWDB data.

## What is not settled

1. **Across the 2017-18 gap.** Fitting three years either side gives jumps of -6 to -14 cm in most units, but the four
   solutions disagree by up to 25 cm in UP, and the jump is also negative in places without pumping (Sri Lanka,
   Maharashtra). Part of the 2016-17 drop is real (it starts before the gap, after the 2014-15 drought monsoons);
   the size of the step is not trustworthy. Trends *within* each mission do not depend on it.
2. **The slowdown since 2018.** Seen in every member for NW India, UP, Rajasthan, Bihar and Bengal. Candidate causes:
   wetter monsoons 2019-2022, Punjab's 2009 law delaying paddy transplanting, canal and recharge works, and a lower base
   after two decades of loss. Distinguishing them needs wells and rainfall-driven modelling (step 05).
3. **Small units.** Punjab-Haryana-Delhi, Sindh, Nepal and Sri Lanka keep less than half their own signal at GRACE
   resolution; report them only inside larger groups, or after downscaling (step 04).
4. **Mountains.** After removing 2000-2019 glacier loss, the Western Himalaya and Karakoram still lose about 1 cm/yr
   since 2018. Glacier loss has accelerated since 2019, so this residual is more likely ice than groundwater.
5. **Irrigation in the land models.** Neither ERA5-Land nor Noah irrigates, so in canal and tube-well areas soil water
   is too low in the dry season. This affects seasonal amplitude more than trends, but it is unquantified.

## How to explain these results

- GRACE weighs water. Each month it measures the change in mass of regions about 300 km across; the groundwater part
  is what is left after subtracting soil water, snow, canopy water and ice, each estimated by models.
- The plains of northwest India and the upper Ganges lost groundwater fastest: about 3-4 cm of water a year in the
  early 2000s over Punjab, Haryana and Delhi, about 11 km3 a year across NW India, roughly the volume of a large
  reservoir every year, drawn by tube wells for rice and wheat.
- The loss has slowed. Since 2018 NW India loses about a fifth of what it lost in 2002-09. That is not yet recovery,
  and we cannot yet say how much is wetter years and how much is policy.
- Central and peninsular India gained storage in good monsoon years; their hard-rock aquifers are small and refill and
  empty with the rains, so their trends follow rainfall.
- Pakistan's Punjab loses steadily at about 1 cm a year. Sindh shows little change: canal seepage recharges it, and
  much of its groundwater is saline.
- Mountain changes are ice, not groundwater.
- Every number is an average over a large area with a range; a flat or slowing average can hide districts where the
  water table keeps falling. Downscaling with wells (step 04) is how those districts will be found.

## Rainfall or people? (src/04_climate_vs_pumping.py, first pass)

Each member's series was fitted with a rainfall memory term (ERA5-Land precipitation anomalies through an exponential
memory of 3-60 months, Humphrey et al. 2017) plus a trend, a mission step and the seasonal cycle; the rainfall-driven
part and the rest were then given their own trends per period (median of the eight members, cm/yr):

| Unit | Rest 2002-09 | Rest 2010-17 | Rest 2018- | Rainfall part 2018- | Memory (months) |
|---|---|---|---|---|---|
| Indian Punjab, Haryana, Delhi | -4.1 | -2.6 | -2.3 | +0.2 | 35 |
| Uttar Pradesh, Uttarakhand | -2.8 | -1.9 | -1.2 | +0.3 | 60 |
| Rajasthan | -2.0 | -2.3 | -0.6 | +0.7 | 60 |
| Pakistan Punjab | -1.4 | -0.7 | -1.0 | -0.1 | 35 |
| Bihar, Jharkhand | -1.5 | -0.5 | +0.3 | 0.0 | 35 |
| Gujarat | -0.2 | +0.8 | +1.8 | +0.4 | 18 |
| South India (hard rock) | -0.7 | -0.1 | +0.6 | +1.0 | 36 |

Reading: in Punjab-Haryana-Delhi and UP the slowdown is in "the rest", not in rainfall: wetter years after 2018 explain
only 0.2-0.3 cm/yr of it. Something about water use or recharge changed (the 2009 paddy-transplanting law, canal
and recharge works, crop or energy changes, or simply a lower water table and fewer working wells). In Rajasthan about
half the recent flattening is rain. In Gujarat the gains are mostly not rain (Narmada canal water and check-dams fit
this). Pakistan's Punjab shows no slowdown. "The rest" is everything the rainfall term does not explain, including
canal seepage and model error, so it is an upper bound on the effect of pumping changes, not a measurement of pumping.
