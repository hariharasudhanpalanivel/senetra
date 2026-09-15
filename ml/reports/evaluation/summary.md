# SENETRA model evaluation

Training run `b873f8f5bba846f38788e287cfc07f0d`, evaluated 2026-09-13 18:42 UTC. Backtest folds whose training window had no outbreak (cold start) are excluded from the headline numbers and reported in their own column.

District WAPE compares models on the same rows. *observed* = current outbreak state assumed to persist (operational default); *scenario* = correct outbreak scenario supplied (what-if / simulator).

| target | level | district WAPE observed | scenario | outbreak (scenario) | cold-start outbreak | XGBoost (central) | local-only | 7-day mean | PHC MAE | PHC noise floor | PHC skill | coverage80 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| patient_footfall | country | 3.8% | 2.5% | 2.2% | 43.8% | 4.1% | 7.7% | 8.8% | 24.353 | 23.483 | 0.113 | 78.1% | promoted to @champion |
| bed_occupancy | country | 2.5% | 1.8% | 1.0% | 30.4% | 2.7% | 4.2% | 5.4% | 8.259 | 7.971 | 0.109 | 76.7% | promoted to @champion |
| staff_availability | country | 0.8% | 0.8% | 0.8% | 0.8% | 0.8% | 1.1% | 0.8% | 5.013 | 5.042 | 0.046 | 79.4% | promoted to @champion |
| ors | country | 4.7% | 3.2% | 3.2% | 50.1% | 5.9% | 11.7% | 10.6% | 13.709 | 13.199 | 0.118 | 79.2% | promoted to @champion |
| paracetamol | country | 4.1% | 2.9% | 3.0% | 45.1% | 5.4% | 7.8% | 9.1% | 19.596 | 18.915 | 0.118 | 77.7% | promoted to @champion |
| iv_fluids | country | 4.1% | 3.3% | 3.6% | 33.6% | 5.0% | 7.7% | 7.2% | 2.926 | 2.892 | 0.067 | 78.8% | promoted to @champion |
| antibiotics | country | 3.1% | 3.1% | 3.0% | 3.0% | 3.1% | 6.2% | 3.3% | 6.257 | 6.291 | 0.046 | 78.0% | promoted to @champion |
| insulin | district | 4.1% | 4.1% | 4.8% | 3.7% | 3.6% | 7.1% | 3.8% | 0.998 | 0.998 | 0.026 | 78.8% | promoted to @champion |

## Gates

- **patient_footfall** v1: district_wape 0.038 <= 0.15 PASS; phc_skill_vs_bl_mean7 0.113 >= 0.0 PASS; district_coverage80 0.781 in [0.7, 0.95] PASS; no_regression_vs_champion 0.038 no other champion skip
- **bed_occupancy** v1: district_wape 0.025 <= 0.15 PASS; phc_skill_vs_bl_mean7 0.109 >= 0.0 PASS; district_coverage80 0.767 in [0.7, 0.95] PASS; no_regression_vs_champion 0.025 no other champion skip
- **staff_availability** v1: district_wape 0.008 <= 0.15 PASS; phc_skill_vs_bl_mean7 0.046 >= 0.0 PASS; district_coverage80 0.794 in [0.7, 0.95] PASS; no_regression_vs_champion 0.008 no other champion skip
- **ors** v1: district_wape 0.047 <= 0.15 PASS; phc_skill_vs_bl_mean7 0.118 >= 0.0 PASS; district_coverage80 0.792 in [0.7, 0.95] PASS; no_regression_vs_champion 0.047 no other champion skip
- **paracetamol** v1: district_wape 0.041 <= 0.15 PASS; phc_skill_vs_bl_mean7 0.118 >= 0.0 PASS; district_coverage80 0.777 in [0.7, 0.95] PASS; no_regression_vs_champion 0.041 no other champion skip
- **iv_fluids** v1: district_wape 0.041 <= 0.15 PASS; phc_skill_vs_bl_mean7 0.067 >= 0.0 PASS; district_coverage80 0.788 in [0.7, 0.95] PASS; no_regression_vs_champion 0.041 no other champion skip
- **antibiotics** v1: district_wape 0.031 <= 0.15 PASS; phc_skill_vs_bl_mean7 0.046 >= 0.0 PASS; district_coverage80 0.780 in [0.7, 0.95] PASS; no_regression_vs_champion 0.031 no other champion skip
- **insulin** v1: district_wape 0.041 <= 0.15 PASS; phc_skill_vs_bl_mean7 0.026 >= 0.0 PASS; district_coverage80 0.788 in [0.7, 0.95] PASS; no_regression_vs_champion 0.041 no other champion skip
