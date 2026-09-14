import pandas as pd

swings = pd.read_csv('pitch_level_deviation_2025.csv', low_memory=False)
print(f"Loaded {len(swings)} swing rows")

# League-wide average deviation per pitch type
by_type = swings.groupby('pitch_type').agg(
    swings                = ('dev_bat_speed', 'count'),
    avg_dev_bat_speed     = ('dev_bat_speed', 'mean'),
    std_dev_bat_speed     = ('dev_bat_speed', 'std'),
    avg_dev_attack_angle  = ('dev_attack_angle', 'mean'),
    avg_dev_swing_length  = ('dev_swing_length', 'mean'),
).reset_index()

by_type['se_bat_speed'] = by_type['std_dev_bat_speed'] / by_type['swings']**0.5

by_type = by_type[by_type['swings'] >= 1000].sort_values('avg_dev_bat_speed', ascending=False)
by_type = by_type.round(3)

by_type.to_csv('deviation_by_pitch_type_2025.csv', index=False)

print("\nLeague-wide average deviation by pitch type (min 1000 swings):")
print(by_type.to_string(index=False))

print("\nPitch type reference: FF=4-seam, SI=sinker, FC=cutter, SL=slider, "
      "ST=sweeper, CU=curveball, KC=knuckle curve, CH=changeup, FS=splitter")