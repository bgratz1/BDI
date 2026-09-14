import pandas as pd
import numpy as np

df = pd.read_csv('statcast_2025_raw.csv', low_memory=False)
baseline = pd.read_csv('batter_baseline_2025.csv')

print(f"Loaded {len(df)} pitches, {len(baseline)} batter+pitch-type baselines")

df = df.copy()
df['is_swing'] = df['description'].isin([
    'swinging_strike', 'swinging_strike_blocked',
    'foul', 'foul_tip', 'hit_into_play'
])
df['is_outside_zone'] = df['zone'].isin([11, 12, 13, 14])
df['is_chase']        = df['is_outside_zone'] & df['is_swing']

# ── PART 1: Swing-level deviation (bat speed + attack direction) ──
# Only swings have bat tracking data
swings = df[df['is_swing']].copy()
print(f"Swings only: {len(swings)} rows")

baseline_swing_cols = baseline[[
    'batter', 'batter_name', 'pitch_type',
    'avg_bat_speed', 'avg_swing_length',
    'avg_attack_angle', 'avg_attack_direction', 'avg_swing_path_tilt',
    'avg_launch_speed',
]]

swings = swings.merge(baseline_swing_cols, on=['batter', 'pitch_type'], how='inner')
print(f"After joining swing baselines: {len(swings)} rows")

# Bat speed deviation: signed, clipped at 0
swings['dev_bat_speed'] = (swings['avg_bat_speed'] - swings['bat_speed']).clip(lower=0)

# Attack direction deviation: signed -- keeps direction information
# Positive = bat path deviated in one direction from baseline
# Negative = bat path deviated in the other direction
# Not taking absolute value so Stage 1 regression can detect
# whether one direction of deviation hurts contact quality
swings['dev_attack_direction'] = (
    swings['attack_direction'] - swings['avg_attack_direction']
)

# Other deviations kept for reference
swings['dev_attack_angle'] = (
    swings['attack_angle'] - swings['avg_attack_angle']
).abs()
swings['dev_swing_path_tilt'] = (
    swings['swing_path_tilt'] - swings['avg_swing_path_tilt']
).abs()

# Z-score within pitch type
for col in ['dev_bat_speed', 'dev_attack_direction',
            'dev_attack_angle', 'dev_swing_path_tilt']:
    means = swings.groupby('pitch_type')[col].transform('mean')
    stds  = swings.groupby('pitch_type')[col].transform('std')
    swings[f'{col}_z'] = (swings[col] - means) / stds

# Save swing-level deviation file
swing_keep_cols = [
    'game_date', 'pitcher', 'player_name', 'pitch_type', 'pitch_name',
    'batter', 'batter_name', 'p_throws', 'stand',
    'release_speed', 'release_spin_rate', 'spin_axis',
    'pfx_x', 'pfx_z', 'release_extension', 'release_pos_x', 'release_pos_z',
    'bat_speed', 'avg_bat_speed', 'dev_bat_speed', 'dev_bat_speed_z',
    'swing_length', 'avg_swing_length',
    'attack_angle', 'avg_attack_angle', 'dev_attack_angle', 'dev_attack_angle_z',
    'swing_path_tilt', 'avg_swing_path_tilt', 'dev_swing_path_tilt', 'dev_swing_path_tilt_z',
    'attack_direction', 'avg_attack_direction', 'dev_attack_direction', 'dev_attack_direction_z',
    'launch_speed', 'avg_launch_speed',
    'description', 'launch_angle',
    'estimated_woba_using_speedangle',
]

swings_out = swings[swing_keep_cols].copy()
swings_out.to_csv('pitch_level_deviation_2025.csv', index=False)
print(f"Saved {len(swings_out)} rows -> pitch_level_deviation_2025.csv")

# ── PART 2: Chase deviation (outside-zone pitches only) ───────────
# Separate from swing mechanics -- only pitches outside the zone
# can generate chase deviation by definition
outside_zone = df[df['is_outside_zone']].copy()
print(f"\nOutside-zone pitches: {len(outside_zone)} rows")

baseline_chase_cols = baseline[[
    'batter', 'pitch_type', 'batter_chase_rate'
]]

outside_zone = outside_zone.merge(
    baseline_chase_cols, on=['batter', 'pitch_type'], how='inner'
)
print(f"After joining chase baselines: {len(outside_zone)} rows")

# Chase deviation per pitch: 1 if batter chased AND their baseline
# chase rate is below league average for this pitch type (i.e. they
# chased when they normally wouldn't), else 0.
# More precisely: actual chase (1/0) minus batter's baseline chase rate
# gives a signed deviation. Positive = chased more than expected.
outside_zone['chase_deviation'] = (
    outside_zone['is_chase'].astype(float) -
    outside_zone['batter_chase_rate']
)
# chase_deviation > 0: batter chased, and they don't normally chase
# this pitch type this often -- pitcher induced an unusual chase
# chase_deviation < 0: batter laid off, and they normally chase more
# chase_deviation = 0: batter behaved exactly as expected

outside_zone_keep_cols = [
    'game_date', 'pitcher', 'player_name', 'pitch_type',
    'batter', 'p_throws', 'stand',
    'is_chase', 'batter_chase_rate', 'chase_deviation',
    'zone', 'plate_x', 'plate_z',
]

outside_zone_out = outside_zone[outside_zone_keep_cols].copy()
outside_zone_out.to_csv('outside_zone_chase_deviation_2025.csv', index=False)
print(f"Saved {len(outside_zone_out)} rows -> outside_zone_chase_deviation_2025.csv")

# ── Preview: pitcher-level chase deviation ────────────────────────
MIN_OUTSIDE_PITCHES = 50
pitcher_chase = outside_zone.groupby(['player_name', 'pitch_type']).agg(
    outside_zone_pitches  = ('is_chase', 'count'),
    avg_chase_deviation   = ('chase_deviation', 'mean'),
    raw_chase_rate        = ('is_chase', 'mean'),
).reset_index()

pitcher_chase = pitcher_chase[
    pitcher_chase['outside_zone_pitches'] >= MIN_OUTSIDE_PITCHES
].sort_values('avg_chase_deviation', ascending=False)

print(f"\nTop 10 pitcher/pitch combos by chase deviation "
      f"(min {MIN_OUTSIDE_PITCHES} outside-zone pitches):")
print(pitcher_chase.head(10).round(3).to_string(index=False))