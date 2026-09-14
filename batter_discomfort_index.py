from pybaseball import statcast, cache
import pandas as pd
import time

cache.enable()

# ── Pull month by month ───────────────────────────────────────
months = [
    ('2025-03-27', '2025-04-30'),
    ('2025-05-01', '2025-05-31'),
    ('2025-06-01', '2025-06-30'),
    ('2025-07-01', '2025-07-31'),
    ('2025-08-01', '2025-08-31'),
    ('2025-09-01', '2025-09-28'),
    ('2025-10-01', '2025-10-31'),
    ('2025-11-01', '2025-11-02'),
]

frames = []
for start, end in months:
    print(f"Pulling {start} to {end}...")
    try:
        df_month = statcast(start_dt=start, end_dt=end)
        frames.append(df_month)
        print(f"  Got {len(df_month)} pitches")
        time.sleep(5)
    except Exception as e:
        print(f"  Failed: {e}, skipping...")

df = pd.concat(frames, ignore_index=True)
print(f"\nTotal pitches: {len(df)}")

df.to_csv('statcast_2025_raw.csv', index=False)
print("Saved to statcast_2025_raw.csv")

# ── Filter to columns we need ─────────────────────────────────
cols = [
    'player_name', 'pitcher', 'pitch_type', 'pitch_name',
    'p_throws', 'game_type', 'balls', 'strikes', 'outs_when_up',
    'release_speed', 'release_spin_rate', 'spin_axis',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'release_extension', 'release_pos_x', 'release_pos_z',
    'description', 'type', 'zone',
    'launch_speed', 'launch_angle',
    'estimated_woba_using_speedangle',
    'estimated_ba_using_speedangle',
    'bb_type', 'events',
]

df = df[cols].copy()

# ── Derived outcome columns ───────────────────────────────────
df['is_swing'] = df['description'].isin([
    'swinging_strike', 'swinging_strike_blocked',
    'foul', 'foul_tip', 'hit_into_play'
])
df['is_whiff']         = df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
df['is_called_strike'] = df['description'] == 'called_strike'
df['is_outside_zone']  = df['zone'].isin([11, 12, 13, 14])
df['is_chase']         = df['is_outside_zone'] & df['is_swing']
df['is_hard_hit']      = df['launch_speed'] >= 95

# ── Aggregate by pitcher + pitch type ────────────────────────
summary = df.groupby(['player_name', 'pitch_type']).agg(
    pitches              = ('pitch_type', 'count'),
    avg_velo             = ('release_speed', 'mean'),
    avg_spin             = ('release_spin_rate', 'mean'),
    avg_pfx_x            = ('pfx_x', 'mean'),
    avg_pfx_z            = ('pfx_z', 'mean'),
    avg_extension        = ('release_extension', 'mean'),
    whiff_rate           = ('is_whiff', 'mean'),
    called_strike_rate   = ('is_called_strike', 'mean'),
    chase_rate           = ('is_chase', 'mean'),
    hard_hit_rate        = ('is_hard_hit', 'mean'),
    xwoba_on_contact     = ('estimated_woba_using_speedangle', 'mean'),
    xba_on_contact       = ('estimated_ba_using_speedangle', 'mean'),
).reset_index()

# Only keep pitch types thrown at least 100 times
summary = summary[summary['pitches'] >= 100].reset_index(drop=True)

summary = summary.round(3)

summary.to_csv('pitch_summary_2025.csv', index=False)

print(f"\nPitcher/pitch combos: {len(summary)}")
print("\nSample:")
print(summary.head(10))