import pandas as pd
import statsmodels.api as sm

swings = pd.read_csv('pitch_level_with_discomfort_score_2025.csv', low_memory=False)
print(f"Loaded {len(swings)} swing rows")

predictors = [
    'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
    'release_extension', 'spin_axis'
]

MIN_SWINGS_PER_TYPE = 1000

pitch_type_counts = swings['pitch_type'].value_counts()
valid_types = pitch_type_counts[pitch_type_counts >= MIN_SWINGS_PER_TYPE].index.tolist()
print(f"\nRunning regressions for pitch types with >= {MIN_SWINGS_PER_TYPE} swings:")
print(pitch_type_counts[valid_types])

results_rows = []

for ptype in valid_types:
    subset = swings[swings['pitch_type'] == ptype].copy()
    subset = subset.dropna(subset=predictors + ['discomfort_score'])

    if len(subset) < MIN_SWINGS_PER_TYPE:
        print(f"\nSkipping {ptype}: only {len(subset)} complete rows after dropping NaNs")
        continue

    X = subset[predictors]
    X = sm.add_constant(X)
    y = subset['discomfort_score']

    model = sm.OLS(y, X).fit()

    print(f"\n{'='*70}")
    print(f"PITCH TYPE: {ptype}  (n = {len(subset)})")
    print(f"{'='*70}")
    print(model.summary())

    for var in predictors:
        results_rows.append({
            'pitch_type': ptype,
            'n': len(subset),
            'variable': var,
            'coef': model.params[var],
            'std_err': model.bse[var],
            'p_value': model.pvalues[var],
            'significant_p05': model.pvalues[var] < 0.05,
            'r_squared': model.rsquared,
        })

results_df = pd.DataFrame(results_rows)
results_df = results_df.round(5)
results_df.to_csv('physics_regression_by_pitch_type_2025.csv', index=False)

print(f"\n\nSaved comparison table -> physics_regression_by_pitch_type_2025.csv")

pivot = results_df.pivot(index='variable', columns='pitch_type', values='coef')
print("\nCoefficient comparison across pitch types:")
print(pivot.round(4).to_string())

sig_pivot = results_df.copy()
sig_pivot['coef_if_sig'] = sig_pivot.apply(
    lambda r: r['coef'] if r['significant_p05'] else None, axis=1
)
sig_view = sig_pivot.pivot(index='variable', columns='pitch_type', values='coef_if_sig')
print("\nSame table, blank = not statistically significant (p >= 0.05):")
print(sig_view.round(4).to_string())