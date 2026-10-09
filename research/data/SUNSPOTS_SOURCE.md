# Annual sunspots replay fixture

`sunspots.csv` is the 309-row annual 1700–2008 series distributed by statsmodels,
originally sourced from the National Geophysical Data Center. Statsmodels marks
this data public domain.

- Data: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/datasets/sunspots/sunspots.csv
- Documentation: https://www.statsmodels.org/v0.13.5/datasets/generated/sunspots.html
- Retrieved 2026-10-09; SHA-256: `f67889b1d9002cd5227f0e0ef54e35b419cdd85a31279adef6f73fb41e5c0a9b`

This is a frozen historical series, not the latest revised solar catalog.
SL-Bench forecasts annual increase/decrease from two preceding direction bits,
warms up before 1800 and evaluates 1800–2008 in chronological test-then-update
order. Values and future labels are never used to initialize the predictor.
Equal consecutive values are encoded as no increase. The transform is fixed,
not fit using the test segment. This is archival streaming replay, not a live
streaming deployment or a physical model of solar activity.
