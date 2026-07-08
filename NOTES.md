# Methodology & Decision Log

A short record of the design choices in this synthetic portfolio project. All
data are generated locally; none of the figures below describe real customers.

## Scope

An end-to-end product-analytics workflow on synthetic digital-banking data:
SQL analysis, classification of an inactivity-based churn label, and a
simulated A/B test of 30-day transaction activity.

## Data generation

- 8,000 synthetic users and 60,000 synthetic transactions, fixed random seed
  for reproducibility.
- Distributions for plan tier, feature adoption and activity are assumed for
  illustration and are not calibrated to any real company's customer base.
- Churn is labelled by inactivity: more than 90 days since the last
  transaction relative to a fixed cutoff (2026-01-01).

## Churn-label classification

- The task is classification of an inactivity-based label at a single
  observation point, not a prospective forecast of future churn.
- Some predictors are derived from the same transaction history that defines
  the label, so they share information with it (documented as a limitation).
- Reported feature importance describes association within the generated
  sample, not causation.

## Simulated A/B test

- The generator injects a higher expected 30-day transaction count into the
  treatment group, so any measured difference is an intended property of the
  data, not evidence about a real product change.
- The analysis reports means, relative difference, a Welch t-test, a 95%
  confidence interval on the mean difference, Cohen's d, and a
  sample-ratio-mismatch check. These demonstrate the statistical-testing
  workflow; they do not justify a real rollout.
- The primary metric is 30-day transaction count, not retention.

## Known limitations

- Synthetic data only; results are not real-world evidence.
- Potential label circularity between churn definition and transaction-based
  features.
- The A/B effect is injected, so significance confirms detection of a known
  difference rather than a genuine intervention effect.

## Possible extensions

- A prospective churn design with separate observation and outcome windows and
  a temporal train/test split.
- Guardrail metrics and a power-analysis example for the experiment section.
