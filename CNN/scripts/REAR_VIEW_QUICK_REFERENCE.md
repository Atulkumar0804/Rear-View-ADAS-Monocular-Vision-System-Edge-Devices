# Rear-View ADAS Quick Reference Guide

## TL;DR - Key Thresholds

| Metric | CRITICAL | WARNING | CAUTION | SAFE |
|--------|----------|---------|---------|------|
| **TTC** | < 1.0s | 1.0-1.5s | 1.5-2.5s | > 2.5s |
| **Distance (high speed)** | < 5m | 5-10m | 10-15m | > 15m |
| **DRAC** | > 3.35 m/s² | 2.0-3.35 m/s² | 0.5-2.0 m/s² | < 0.5 m/s² |
| **PET** | < 1.0s | 1.0-2.0s | 2.0-3.0s | > 3.0s |

## Color Coding
- 🔴 **RED (0,0,255)**: CRITICAL - IMMEDIATE DANGER
- 🟠 **ORANGE (0,165,255)**: WARNING - HIGH RISK
- 🟡 **YELLOW (0,255,255)**: CAUTION - MONITOR
- 🟢 **GREEN (0,255,0)**: SAFE - OK

## What Each SSM Means

### TTC (Time to Collision)
"In how many seconds will the rear vehicle hit me if nothing changes?"
- **< 1.0s**: Collision in less than 1 second → CRITICAL
- **1.0-2.5s**: Potential collision → WARNING
- **> 2.5s**: Safe time → OK

**Example**: Distance 20m, your speed 50km/h (13.9 m/s), rear vehicle 70km/h (19.4 m/s)
- Relative speed = 19.4 - 13.9 = 5.5 m/s
- TTC = 20 / 5.5 = 3.6 seconds ✓ SAFE

### DRAC (Deceleration to Avoid Collision)
"How hard does the rear vehicle need to brake to avoid hitting me?"
- **> 3.35 m/s²**: Beyond normal braking capacity → CRITICAL
- **0.5-3.35 m/s²**: Normal to hard braking → WARNING/SAFE
- **< 0.5 m/s²**: Gentle braking → SAFE

**Example**: Same scenario above, DRAC ≈ 0.75 m/s² ✓ SAFE

### PET (Post Encroachment Time)
"After I leave, how long until the rear vehicle reaches my position?"
- **< 1.0s**: Less than 1 second margin → CRITICAL
- **> 3.0s**: Safe margin → OK

### MTTC (Modified TTC)
Like TTC but accounts for vehicles changing speed.
- Used when rear vehicle is accelerating/braking
- Better for real-world conditions

## Safety Alert Types

### collision_imminent
🔴 **ACTION REQUIRED**
- Multiple critical SSMs triggered
- Immediate deceleration needed
- Audio/visual alert
- Example: TTC < 1.0s AND DRAC > 3.35

### collision_warning
🟠 **WARNING - Prepare to brake**
- TTC < 1.5s
- Significant collision risk
- Monitor closely
- Example: TTC = 1.2s

### high_deceleration
🟠 **WARNING - Check rear**
- Rear vehicle needs hard braking
- DRAC > 2.0 m/s²
- Example: DRAC = 2.5 m/s²

### distance_warning
🟡 **CAUTION - Close following**
- Distance < 15m at highway speed
- Normal operation but monitor
- Example: Distance = 12m

## Use Cases & Scenarios

### Scenario 1: Clear Road Behind
```
Detections: 0
Safety Level: SAFE
Color: GREEN
Action: Normal driving
```

### Scenario 2: Car Maintaining Distance
```
Distance: 25m
Your Speed: 80 km/h
Rear Speed: 80 km/h
TTC: Infinity (same speed)
Safety Level: SAFE
Color: GREEN
```

### Scenario 3: Car Approaching Gradually
```
Distance: 20m
Your Speed: 80 km/h (22.2 m/s)
Rear Speed: 90 km/h (25 m/s)
TTC: 20 / (25-22.2) = 6.9s
Safety Level: CAUTION
Color: YELLOW
Action: Maintain awareness
```

### Scenario 4: Car Approaching Rapidly (DANGEROUS!)
```
Distance: 15m
Your Speed: 80 km/h (22.2 m/s)
Rear Speed: 110 km/h (30.5 m/s)
TTC: 15 / (30.5-22.2) = 1.8s
Safety Level: WARNING
Color: ORANGE
Action: Prepare to brake/signal
```

### Scenario 5: Emergency - Imminent Collision
```
Distance: 5m
Your Speed: 50 km/h (13.9 m/s)
Rear Speed: 100 km/h (27.8 m/s)
TTC: 5 / (27.8-13.9) = 0.35s
DRAC: 27.8² / (2×5) = 77.2 m/s² (!!)
Safety Level: CRITICAL
Color: RED
Action: EMERGENCY BRAKING
```

## How to Interpret the Video Output

### Bounding Box
```
┌─────────────────────────────┐
│ Sedan: 0.95 | D:15.2m      │ ← Class & Distance
│ C:15.2 ML:14.8             │ ← Depth estimates
│  ┌─────────────────────┐    │
│  │                     │    │
│  │  [ORANGE] collision_warning ← Safety Level
│  │  TTC:1.35s DRAC:2.5m/s²    ← SSM metrics
│  │                     │    │ 
│  │   APPROACHING       │    │ ← Motion State
│  │   25.5 km/h         │    │ ← Velocity
│  └─────────────────────┘    │
│        15.2m                │ ← Large distance display
└─────────────────────────────┘
```

### Top-Left Info
```
Scenario: approaching_vehicle  ← What's happening
Threat: medium                 ← Overall threat level
Critical Vehicles: 1           ← How many dangerous cars
```

## CSV Column Guide

For analyzing recorded data:

```
frame_number     | Video frame (use with FPS for time)
track_id         | Vehicle tracking ID
vehicle_class    | Car type detected
distance_m       | Distance in meters
motion_state     | approaching/receding/stable
speed_kmh        | Vehicle speed

SAFETY COLUMNS:
safety_level     | CRITICAL/WARNING/CAUTION/SAFE
alert_type       | Type of alert (if any)
ttc_s            | Time to Collision (seconds)
mttc_s           | Modified TTC (with acceleration)
drac_ms2         | Required deceleration (m/s²)
scenario_type    | Scene type (clear_rear/approaching_vehicle/etc)
```

## Quick Decision Matrix

### Decision Tree:
```
Is vehicle approaching?
├─ NO → Safety: SAFE ✓
└─ YES
   ├─ Distance > 20m?
   │  ├─ YES → TTC check
   │  │       ├─ TTC > 2.5s → SAFE ✓
   │  │       ├─ TTC 1.5-2.5s → CAUTION (Yellow)
   │  │       └─ TTC < 1.5s → WARNING (Orange)
   │  └─ NO → Check DRAC
   │         ├─ DRAC < 2.0 → CAUTION
   │         ├─ DRAC 2.0-3.35 → WARNING
   │         └─ DRAC > 3.35 → CRITICAL (Red)
   └─ Multiple critical SSMs?
      └─ YES → CRITICAL (Red) 🚨
```

## Common Issues & Solutions

### False Positives (Red alert when safe)
- **Cause**: Detection error or false vehicle
- **Solution**: Ignore if distance suddenly increases

### Delayed Alerts
- **Cause**: Processing lag or tracking delay
- **Solution**: Use smooth Kalman filter (already in system)

### No Alert (Safety = Unknown)
- **Cause**: Distance not estimated
- **Solution**: Ensure vehicle is clearly visible

### Flickering Color
- **Cause**: Multiple vehicles or tracking loss
- **Solution**: Visual stability improves after few frames

## Data Export for Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('result_detections.csv')

# Timeline of critical events
critical = df[df['safety_level'] == 'CRITICAL']
print(f"Total critical events: {len(critical)}")
print(f"Timestamps: {critical['timestamp_s'].tolist()}")

# Average TTC by vehicle class
ttc_analysis = df.groupby('vehicle_class')['ttc_s'].mean()
print("\nAverage TTC by vehicle class:")
print(ttc_analysis)

# Plot distance over time
df.plot(x='timestamp_s', y='distance_m', figsize=(12,6))
plt.title('Distance vs Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Distance (meters)')
plt.axhline(y=10, color='r', linestyle='--', label='Critical Distance')
plt.legend()
plt.show()
```

## Performance Tips

### For Real-time Processing
1. Use `--device cuda` for GPU acceleration
2. Adjust `--zoedepth-interval 30` if slow
   - Higher = faster, less accurate
   - Lower = slower, more accurate

### For Analysis
1. Process once, analyze CSV multiple times
2. Filter by safety_level for events of interest
3. Export specific time ranges for detailed review

## Emergency Response Guide

### If CRITICAL Alert (Red):
1. **Check mirrors** - Verify visually
2. **Slow down gradually** - Don't brake hard unnecessarily
3. **Signal intentions** - Turn signal, brake lights
4. **Move to side** - Make space if possible
5. **Be prepared** - Ready for emergency braking

### If WARNING Alert (Orange):
1. **Increase awareness** - Monitor constantly
2. **Reduce speed slightly** - Create more margin
3. **Prepare brake** - Move foot to brake pedal
4. **Plan maneuver** - Consider lane change if safe

### If CAUTION Alert (Yellow):
1. **Monitor** - Keep checking mirrors
2. **Maintain speed** - Don't accelerate
3. **Anticipate** - Expect lane change or braking

## Configuration Quick Reference

```bash
# Standard dash cam use
python video_inference.py --input video.mp4 --output result.mp4 --device cuda

# Fast processing (less accurate depth)
python video_inference.py --input video.mp4 --output result.mp4 --zoedepth-interval 60

# High accuracy (slower processing)
python video_inference.py --input video.mp4 --output result.mp4 --zoedepth-interval 15 --alpha-lr 0.02

# CPU only (very slow, no GPU)
python video_inference.py --input video.mp4 --output result.mp4 --device cpu
```

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| All green, no alerts | Detector not working | Check video quality |
| Constant red alerts | False detections | Check for artifacts |
| Crashes on large video | Out of memory | Process in chunks |
| Very slow processing | CPU only | Use `--device cuda` |
| Distance always wrong | Camera calibration off | Recalibrate FOV |

## Related Files

- Main: `/inference/video_inference.py`
- Config: `REAR_VIEW_SAFETY_ASSESSMENT.md`
- Logs: `*_detections.csv`
- Models: `/models/` directory

## Support & References

For detailed information see: `REAR_VIEW_SAFETY_ASSESSMENT.md`

For research context: Traffic safety evaluation using surrogate safety measures (2025)
- Authors: N. Mohamed Hasain, Mokaddes Ali Ahmed
- Journal: IATSS Research, Volume 49

---

**Quick Tips:**
- 🟢 Green = Safe, keep driving normally
- 🟡 Yellow = Be careful, monitor situation
- 🟠 Orange = Warning, prepare to react
- 🔴 Red = EMERGENCY, take action immediately

**Remember:** These are automated alerts. Always use judgment and confirm with visual checks!
