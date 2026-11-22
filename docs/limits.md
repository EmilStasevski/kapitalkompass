# Limit system and escalation rule

Configured in `config/steering.yaml`; computed by `src/kapitalkompass/steering/limits.py`;
results in `results/steering/limits_before.csv`, `limits_after.csv` (after re-allocation) and
`limits_after_actions.csv` (after re-allocation, IRRBB hedge and single-name sell-down).
All limits are internal bank policy (assumptions), except the EVE limit, which uses the
BCBS 368 para 88 outlier level (15% of Tier 1).

## Risk-bearing capacity

Risk cover (Risikodeckungspotenzial) = CET1 equity. Credit risk uses economic capital
(ES₉₉.₉ − EL, module 1); interest-rate risk uses the worst ΔEVE loss over the six BCBS 368
shocks (module 3). Total risk = credit EC + IRRBB loss, without diversification between
risk types (conservative).

| Limit | Exposure measure | Limit |
|---|---|---|
| credit_risk_capital | Credit EC | 45% of risk cover |
| irrbb_eve | max(0, −worst ΔEVE) | 15% of Tier 1 |
| total_risk_cover | Credit EC + IRRBB loss | 80% of risk cover |
| nii_shock | max(0, −worst ΔNII) | 5% of Tier 1 |
| sector_concentration | Largest sector EAD share | 45% |
| single_name | Largest single-name EAD / CET1 | 15% |
| lcr | LCR | green ≥ 130%, amber 110–130%, red < 110% |

## Traffic lights

Utilisation u = exposure / limit (for the LCR: floor / LCR).

* **green**: u < 80%
* **amber**: 80% ≤ u ≤ 100% (early warning, includes "at the limit")
* **red**: u > 100% (limit breach)

## Escalation

| Status | Rule |
|---|---|
| green | No action. Monthly reporting to the ALCO / risk committee. |
| amber | Early warning: risk controlling informs the risk committee within 5 business days; the business owner submits an action plan within 20 business days. |
| red | Limit breach: immediate ad-hoc report to the management board (Vorstand); measures to restore the limit decided within 10 business days and implemented within 30; documented in the next risk report. |

## Management actions computed by the pipeline

1. Re-allocation of segment volumes (cvxpy LP, `results/steering/optimisation.json`).
2. IRRBB hedge: smallest receive- or pay-fixed 5y swap notional that brings the worst ΔEVE
   to 90% of the EVE limit across all six scenarios (LP over the notional,
   `results/steering/hedge.json`, including the hedge's ΔNII effect).
3. Single-name sell-down to the 15%-of-CET1 limit (`results/steering/actions.json`).
