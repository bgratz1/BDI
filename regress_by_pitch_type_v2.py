import pandas as pd
import statsmodels.api as sm
import numpy as np

swings = pd.read_csv('pitch_level_with_discomfort_score_2025.csv', low_memory=False)
print(f"Loaded {len(swings)} swing rows")

# Fix 1: handedness-adjust pfx_x
swings = swings.copy()
swings['pfx_x_adj'] = np.where(
    swings['p_throws'] == 'R',
    swings['pfx_x'],
    -swings['pfx_x']
)

predictors = [
    'release_speed', 'release_spin_rate', 'pfx_x_adj', 'pfx_z',
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

    # Pairwise correlation check among predictors
    corr_matrix = subset[predictors].corr()
    print(f"\n{'='*70}")
    print(f"PITCH TYPE: {ptype}  (n = {len(subset)}) -- predictor correlations")
    print(f"{'='*70}")
    print(corr_matrix.round(2).to_string())

    high_corr_pairs = []
    for i, var1 in enumerate(predictors):
        for var2 in predictors[i+1:]:
            r = corr_matrix.loc[var1, var2]
            if abs(r) > 0.6:
                high_corr_pairs.append((var1, var2, round(r, 2)))
    if high_corr_pairs:
        print(f"  ! High correlation pairs (|r| > 0.6): {high_corr_pairs}")
    else:
        print("  No predictor pairs with |r| > 0.6")

    # Fix 2: standardize predictors (z-score)
    X_raw = subset[predictors]
    X_z = (X_raw - X_raw.mean()) / X_raw.std()
    X_z = sm.add_constant(X_z)
    y = subset['discomfort_score']

    model = sm.OLS(y, X_z).fit()

    print(f"\nOLS RESULTS (standardized predictors): {ptype}")
    print(model.summary())
    print(f"Condition number: {model.condition_number:.2f}")

    for var in predictors:
        results_rows.append({
            'pitch_type': ptype,
            'n': len(subset),
            'variable': var,
            'std_coef': model.params[var],
            'std_err': model.bse[var],
            'p_value': model.pvalues[var],
            'significant_p05': model.pvalues[var] < 0.05,
            'r_squared': model.rsquared,
            'condition_number': model.condition_number,
        })

results_df = pd.DataFrame(results_rows)
results_df = results_df.round(5)
results_df.to_csv('physics_regression_standardized_2025.csv', index=False)
print(f"\n\nSaved -> physics_regression_standardized_2025.csv")

pivot = results_df.pivot(index='variable', columns='pitch_type', values='std_coef')
print("\nStandardized coefficient comparison across pitch types:")
print("(each value = effect on discomfort_score per 1 SD change in that variable)")
print(pivot.round(4).to_string())

sig_pivot = results_df.copy()
sig_pivot['coef_if_sig'] = sig_pivot.apply(
    lambda r: r['std_coef'] if r['significant_p05'] else None, axis=1
)
sig_view = sig_pivot.pivot(index='variable', columns='pitch_type', values='coef_if_sig')
print("\nSame table, blank = not statistically significant (p >= 0.05):")
print(sig_view.round(4).to_string())

cond_numbers = results_df.groupby('pitch_type')['condition_number'].first()
print("\nCondition numbers by pitch type (compare to first-pass values, should be much lower):")
print(cond_numbers.round(2).to_string())