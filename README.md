# swTideModel
A barotropic shallow water model with equilibrium and self-attraction &amp; loading tidal forces

This is a single-layer, dissipative, non-rotating shallow
water model with tidal equilibrium forcing, tidal self-attraction and
loading (SAL) forcing, and wind forcing.
This model will be used to test numerical algorithms for applying SAL
terms efficiently, using predictor and predictor-corrector schemes,
with the goal of decreasing how often they need to be calculated explicitly.

The current model is well-commented, but not well-documented.
A future version will include a more user-friendly interface and better documentation (e.g., fundamental equations, assumptions, boundary options documentation).
