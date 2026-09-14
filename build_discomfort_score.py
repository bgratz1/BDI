import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load both deviation files
swings = pd.read_csv('pitch_level_deviation_2025.csv', low_memory=False)
chase  = pd.read_csv('outside_zone_chase_deviation_2025.csv', low_memory=False)
print(f"Loaded {len(swings)} swing rows, {len(chase)} outside-zone rows")

swings['is_whiff'] = swings['description'].isin(
    ['swinging_strike', 'swinging_strike_blocked']
).astype(int)

# ════════════════════════════════════════════════════════════════
# STAGE 1: OLS on launch speed (contact swings only)
# Trains physics score weights on contact quality.
# Coefficients will be negative (more deviation = lower launch speed)
# so sign is flipped when building the score.
# ════════════════════════════════════════════════════════════════
dev_cols = [
    'dev_bat_speed_z',
    'dev_attack_direction_z',
    'dev_swing_path_tilt_z',
]

contact = swings[swings['launch_speed'].notna()].copy()
contact = contact.dropna(subset=dev_cols + ['launch_speed'])
print(f"\nContact swings for Stage 1 regression: {len(contact)}")

X1 = contact[dev_cols]
X1 = sm.add_constant(X1)
y1 = contact['launch_speed']

stage1_model = sm.OLS(y1, X1).fit()
print("\n" + "="*60)
print("STAGE 1: OLS ON LAUNCH SPEED (contact swings only)")
print("launch_speed ~ dev_bat_speed_z + dev_attack_direction_z")
print("="*60)
print(stage1_model.summary())

coefs = stage1_model.params.drop('const')
print("\nCoefficients (expect negative):")
print(coefs)

wrong_sign = coefs[coefs > 0]
if len(wrong_sign) > 0:
    print(f"\nWARNING: positive coefficients found: {list(wrong_sign.index)}")
else:
    print("\nBoth coefficients negative as expected.")

# Build physics score on ALL swings (sign flipped)
swings_scored = swings.dropna(subset=dev_cols).copy()
swings_scored['is_whiff'] = swings_scored['description'].isin(
    ['swinging_strike', 'swinging_strike_blocked']
).astype(int)

swings_scored['physics_score'] = -(
    coefs['dev_bat_speed_z']        * swings_scored['dev_bat_speed_z'] +
    coefs['dev_attack_direction_z'] * swings_scored['dev_attack_direction_z']
)

print(f"\nPhysics score built for {len(swings_scored)} swings")
print(swings_scored['physics_score'].describe())

swings_scored.to_csv('pitch_level_with_discomfort_score_2025.csv', index=False)
print("Saved -> pitch_level_with_discomfort_score_2025.csv")

# Aggregate physics to pitcher/pitch-type level
MIN_SWINGS = 100

pitcher_physics = swings_scored.groupby(['player_name', 'pitch_type']).agg(
    swings           = ('physics_score', 'count'),
    avg_physics      = ('physics_score', 'mean'),
    avg_xwoba        = ('estimated_woba_using_speedangle', 'mean'),
    whiff_rate       = ('is_whiff', 'mean'),
    avg_launch_speed = ('launch_speed', 'mean'),
).reset_index()

pitcher_physics = pitcher_physics[
    pitcher_physics['swings'] >= MIN_SWINGS
].reset_index(drop=True)

print(f"\nAggregated physics: {len(pitcher_physics)} pitcher/pitch-type combos")

# Aggregate chase deviation to pitcher/pitch-type level
MIN_OUTSIDE = 50

pitcher_chase = chase.groupby(['player_name', 'pitch_type']).agg(
    outside_zone_pitches = ('is_chase', 'count'),
    avg_chase_deviation  = ('chase_deviation', 'mean'),
    raw_chase_rate       = ('is_chase', 'mean'),
).reset_index()

pitcher_chase = pitcher_chase[
    pitcher_chase['outside_zone_pitches'] >= MIN_OUTSIDE
].reset_index(drop=True)

print(f"Aggregated chase: {len(pitcher_chase)} pitcher/pitch-type combos")

# Merge
combined = pitcher_physics.merge(
    pitcher_chase[['player_name', 'pitch_type', 'avg_chase_deviation',
                   'outside_zone_pitches', 'raw_chase_rate']],
    on=['player_name', 'pitch_type'],
    how='inner'
).dropna(subset=['avg_physics', 'avg_chase_deviation', 'avg_xwoba'])

print(f"Combined: {len(combined)} pitcher/pitch-type combos")

# Z-score both for Stage 2
combined['physics_z'] = (
    (combined['avg_physics'] - combined['avg_physics'].mean()) /
    combined['avg_physics'].std()
)
combined['chase_z'] = (
    (combined['avg_chase_deviation'] - combined['avg_chase_deviation'].mean()) /
    combined['avg_chase_deviation'].std()
)

# ════════════════════════════════════════════════════════════════
# STAGE 2: OLS on whiff rate (pitcher/pitch-type level)
# Different outcome from Stage 1 (launch speed) and from
# validation (xwOBA). Both coefficients should be positive.
# ════════════════════════════════════════════════════════════════
X2 = sm.add_constant(combined[['physics_z', 'chase_z']])
y2 = combined['whiff_rate']

stage2_model = sm.OLS(y2, X2).fit()
print("\n" + "="*60)
print("STAGE 2: OLS ON WHIFF RATE (pitcher/pitch-type level)")
print("whiff_rate ~ physics_z + chase_z")
print("="*60)
print(stage2_model.summary())

physics_coef = stage2_model.params['physics_z']
chase_coef   = stage2_model.params['chase_z']
print(f"\nPhysics coefficient:         {physics_coef:.4f}  (expect positive)")
print(f"Chase deviation coefficient: {chase_coef:.4f}  (expect positive)")

if physics_coef < 0:
    print("WARNING: physics coefficient negative -- unexpected")
if chase_coef < 0:
    print("WARNING: chase coefficient negative -- unexpected")

# Build final combined discomfort score
combined['discomfort_score'] = (
    physics_coef * combined['physics_z'] +
    chase_coef   * combined['chase_z']
)

print(f"\nFinal discomfort score distribution:")
print(combined['discomfort_score'].describe())

# ════════════════════════════════════════════════════════════════
# VALIDATION: xwOBA -- fully held-out
# Stage 1 used launch speed, Stage 2 used whiff rate.
# xwOBA was never used anywhere in training.
# ════════════════════════════════════════════════════════════════
for outcome, label in [
    ('avg_xwoba',        'avg xwOBA allowed'),
    ('avg_launch_speed', 'avg launch speed allowed'),
]:
    subset = combined.dropna(subset=['discomfort_score', outcome])
    X_val  = sm.add_constant(subset['discomfort_score'])
    y_val  = subset[outcome]
    model  = sm.OLS(y_val, X_val).fit()
    corr   = subset['discomfort_score'].corr(subset[outcome])

    print(f"\n{'='*60}")
    print(f"VALIDATION (held-out): {label} ~ discomfort_score")
    print(f"{'='*60}")
    print(f"n          = {len(subset)}")
    print(f"R²         = {model.rsquared:.4f}")
    print(f"Coef       = {model.params['discomfort_score']:.4f}  "
          f"(p = {model.pvalues['discomfort_score']:.4f})")
    print(f"Correlation= {corr:.4f}")

# Save leaderboard
combined = combined.round(4)
combined.to_csv('discomfort_leaderboard_2025.csv', index=False)

print(f"\nTop 15 by final discomfort score:")
print(
    combined.sort_values('discomfort_score', ascending=False)
    .head(15)[[
        'player_name', 'pitch_type', 'swings',
        'outside_zone_pitches', 'avg_physics',
        'avg_chase_deviation', 'discomfort_score',
        'avg_xwoba', 'whiff_rate'
    ]]
    .to_string(index=False)
)