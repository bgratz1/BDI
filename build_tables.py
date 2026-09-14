import pandas as pd

df = pd.read_csv('statcast_2025_raw.csv', low_memory=False)
print(f"Loaded {len(df)} pitches")

df = df.copy()
df['is_swing'] = df['description'].isin([
    'swinging_strike', 'swinging_strike_blocked',
    'foul', 'foul_tip', 'hit_into_play'
])
df['is_whiff']         = df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
df['is_called_strike']  = df['description'] == 'called_strike'
df['is_outside_zone']   = df['zone'].isin([11, 12, 13, 14])
df['is_chase']          = df['is_outside_zone'] & df['is_swing']
df['is_hard_hit']       = df['launch_speed'] >= 95

# TABLE 1: Pitcher pitch summary
pitch_summary = df.groupby(['player_name', 'pitcher', 'pitch_type']).agg(
    pitches              = ('pitch_type', 'count'),
    avg_velo             = ('release_speed', 'mean'),
    avg_spin             = ('release_spin_rate', 'mean'),
    avg_spin_axis        = ('spin_axis', 'mean'),
    avg_pfx_x            = ('pfx_x', 'mean'),
    avg_pfx_z            = ('pfx_z', 'mean'),
    avg_extension        = ('release_extension', 'mean'),
    avg_release_pos_x    = ('release_pos_x', 'mean'),
    avg_release_pos_z    = ('release_pos_z', 'mean'),
    whiff_rate           = ('is_whiff', 'mean'),
    called_strike_rate   = ('is_called_strike', 'mean'),
    chase_rate           = ('is_chase', 'mean'),
    hard_hit_rate        = ('is_hard_hit', 'mean'),
    xwoba_on_contact     = ('estimated_woba_using_speedangle', 'mean'),
    xba_on_contact       = ('estimated_ba_using_speedangle', 'mean'),
).reset_index()

pitch_summary = pitch_summary[pitch_summary['pitches'] >= 100].reset_index(drop=True)
pitch_summary = pitch_summary.round(3)
pitch_summary.to_csv('pitch_summary_2025.csv', index=False)
print(f"Pitcher pitch summary: {len(pitch_summary)} rows -> pitch_summary_2025.csv")

# TABLE 2: Batter baseline (per batter + pitch type)
batter_baseline = df.groupby(['batter', 'pitch_type']).agg(
    pitches_seen          = ('batter', 'count'),
    swings                = ('is_swing', 'sum'),
    avg_bat_speed         = ('bat_speed', 'mean'),
    avg_swing_length      = ('swing_length', 'mean'),
    avg_attack_angle      = ('attack_angle', 'mean'),
    avg_attack_direction  = ('attack_direction', 'mean'),
    avg_swing_path_tilt   = ('swing_path_tilt', 'mean'),
    whiff_rate            = ('is_whiff', 'mean'),
    called_strike_rate    = ('is_called_strike', 'mean'),
    hard_hit_rate         = ('is_hard_hit', 'mean'),
    contact_swings        = ('launch_speed', 'count'),
    avg_launch_speed      = ('launch_speed', 'mean'),
    avg_launch_angle      = ('launch_angle', 'mean'),
    avg_xwoba             = ('estimated_woba_using_speedangle', 'mean'),
    outside_zone_pitches  = ('is_outside_zone', 'sum'),
    chases                = ('is_chase', 'sum'),
).reset_index()

# Chase rate = chases / outside_zone_pitches
# Using explicit division rather than mean on is_chase so the
# denominator is only outside-zone pitches, not all pitches
batter_baseline['batter_chase_rate'] = (
    batter_baseline['chases'] /
    batter_baseline['outside_zone_pitches'].clip(lower=1)
)

batter_names = df.groupby('batter')['player_name'].agg(
    lambda x: x.mode().iloc[0]
).reset_index()
batter_names = batter_names.rename(columns={'player_name': 'batter_name'})
batter_baseline = batter_baseline.merge(batter_names, on='batter', how='left')

cols = ['batter', 'batter_name', 'pitch_type'] + [
    c for c in batter_baseline.columns
    if c not in ('batter', 'batter_name', 'pitch_type')
]
batter_baseline = batter_baseline[cols]

# Minimum 30 swings AND 20 outside-zone pitches for reliable baselines
MIN_SWINGS       = 30
MIN_OUTSIDE_ZONE = 20
batter_baseline = batter_baseline[
    (batter_baseline['swings'] >= MIN_SWINGS) &
    (batter_baseline['outside_zone_pitches'] >= MIN_OUTSIDE_ZONE)
].reset_index(drop=True)

batter_baseline = batter_baseline.round(3)
batter_baseline.to_csv('batter_baseline_2025.csv', index=False)
print(f"Batter baseline: {len(batter_baseline)} rows -> batter_baseline_2025.csv")
print(f"  ({batter_baseline['batter'].nunique()} unique batters, "
      f"{batter_baseline['pitch_type'].nunique()} pitch types)")
print(f"\nSample chase rates by pitch type (league averages):")
print(
    batter_baseline.groupby('pitch_type')['batter_chase_rate']
    .mean().round(3).sort_values(ascending=False).to_string()
)