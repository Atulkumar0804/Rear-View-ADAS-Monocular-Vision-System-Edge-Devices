# Lane-Aware Safety Assessment Guide

## Overview

The updated rear-view ADAS now generates **lane-aware warnings** that differentiate between vehicles in the **same lane** (collision risk) and **adjacent lanes** (awareness only). This prevents false alarms from vehicles in other lanes while providing critical collision warnings for same-lane traffic.

---

## How Lane Detection Works

### Lane Division

The frame is divided into **3 vertical zones**:

```
[LEFT LANE]  |  [CENTER/SAME LANE]  |  [RIGHT LANE]
  0-33%      |        33-66%         |    66-100%
```

### Lane Assignment

- **Bounding box center position** determines primary lane
- **Vehicle coverage analysis** considers if vehicle spans multiple lanes
- **Confidence score** indicates how centered the vehicle is in its detected lane

---

## Warning Levels by Lane

### SAME LANE VEHICLES (Center/Collision Risk)

| Safety Level | Color | TTC Threshold | Rider Action | Example |
|---|---|---|---|---|
| **CRITICAL** | 🔴 Red | < 1.0s | ⚠️ IMMEDIATE BRAKE | "Apply strong brakes immediately!" |
| **WARNING** | 🟠 Orange | 1.0-1.5s | 🔻 Strong Decelerate | "Decelerate aggressively!" |
| **CAUTION** | 🟡 Yellow | 1.5-2.5s | 📉 Monitor | "Reduce speed gradually" |
| **SAFE** | 🟢 Green | > 2.5s | ✓ Maintain | "Continue normal driving" |

---

### ADJACENT LANE VEHICLES (Left/Right - Awareness Only)

| Lane | Color | Message | Rider Action | Example |
|---|---|---|---|---|
| **LEFT LANE** | 🟦 Blue-ish | "Vehicle in LEFT lane - Xm away" | 👁️ Monitor | "Monitor vehicle in left lane" |
| **RIGHT LANE** | 🟨 Yellow-ish | "Vehicle in RIGHT lane - Xm away" | 👁️ Monitor | "Monitor vehicle in right lane" |

**Note:** Adjacent lane vehicles generated **INFO** level alerts - they will NOT trigger collision warnings because they cannot impact your lane directly.

---

## Rider Action Recommendations

### Action Types

```
┌─────────────────────────────────────────────────────┐
│ SAME LANE (Collision Risk)                          │
├─────────────────────────────────────────────────────┤
│ ▶ EMERGENCY_BRAKE (CRITICAL)                        │
│   → "Apply strong brakes immediately!"              │
│   → When: Vehicle < 0.5s away & approaching fast    │
│                                                      │
│ ▶ STRONG_DECELERATE (WARNING)                       │
│   → "Decelerate aggressively!"                      │
│   → When: Vehicle < 1.0s away                       │
│                                                      │
│ ▶ DECELERATE (WARNING)                              │
│   → "Reduce speed gradually"                        │
│   → When: Vehicle 1.0-1.5s away                     │
│                                                      │
│ ▶ MONITOR (CAUTION)                                 │
│   → "Monitor distance"                              │
│   → When: 1.5-2.5s away                             │
│                                                      │
│ ▶ MAINTAIN_SPEED (SAFE)                             │
│   → "Continue normal driving"                       │
│   → When: > 2.5s away (safe distance)               │
├─────────────────────────────────────────────────────┤
│ ADJACENT LANES (No Collision Risk)                  │
├─────────────────────────────────────────────────────┤
│ ▶ BE_AWARE                                          │
│   → "Monitor vehicle in [LANE]"                     │
│   → When: Adjacent lane vehicle < 15m               │
│                                                      │
│ ▶ MONITOR                                           │
│   → "Stay aware of surroundings"                    │
│   → When: Adjacent lane vehicle > 15m               │
└─────────────────────────────────────────────────────┘
```

---

## Screen Visualization

### Example Output 1: Same-Lane Critical Warning

```
┌────────────────────────────────────┐
│  [CRITICAL] collision_imminent      │   ← RED box (CRITICAL)
│  TTC: 0.95s DRAC: 3.8m/s²         │
│  → Apply strong brakes immediately! │   ← RED action instruction
│                                      │
│  Distance: 9.2m                     │
│  Speed: 48km/h (APPROACHING)        │
└────────────────────────────────────┘
```

### Example Output 2: Adjacent Lane Information

```
┌────────────────────────────────────┐
│  [RIGHT LANE] Vehicle 17.5m away    │   ← ORANGE box (INFO)
│  → Monitor vehicle in right lane.   │   ← ORANGE action instruction
│                                      │
│  Distance: 17.5m                    │
│  Speed: 35km/h (STABLE)             │
└────────────────────────────────────┘
```

---

## Key Differences from Previous System

| Aspect | Previous | New Lane-Aware |
|---|---|---|
| **All Lane Vehicles** | Same collision warnings | Different warnings per lane |
| **Adjacent Lane Vehicles** | RED CRITICAL warnings | Blue/Orange INFO only |
| **False Alarms** | High (warnings for safe adjacent vehicles) | Low (no collision warnings except same lane) |
| **Rider Guidance** | Generic safety level | Action-specific instructions ("brake", "monitor", etc.) |
| **Lane Awareness** | Not considered | Primary decision factor |

---

## How to Read the Visualization

### Color Key

```
🔴 RED    = CRITICAL same-lane collision risk → BRAKE NOW
🟠 ORANGE = WARNING same-lane approaching → DECELERATE
🟡 YELLOW = CAUTION same-lane distance → MONITOR  
🟢 GREEN  = SAFE same-lane distance → MAINTAIN
🔵 BLUE   = INFO adjacent lane vehicle → MONITOR ONLY
```

### Example Frame Analysis

Your rear-view camera detects multiple vehicles:

```
Vehicle 1 (Center):    RED box with "TTC: 0.8s" → CRITICAL (same lane)
Vehicle 2 (Left):      ORANGE box "LEFT LANE" → MONITOR (adjacent)
Vehicle 3 (Right):     BLUE box "RIGHT LANE" → MONITOR (adjacent)
```

- **Decision**: Focus on Vehicle 1 (same lane collision risk)
- **Ignore**: Vehicles 2 & 3 (different lanes, no collision impact)

---

## Rider Decision Making

### Quick Reference Table (Rider Action Mapping)

```
┌──────────────┬────────────────────┬─────────────────────┐
│ Situation    │ Safety Level       │ Rider Action        │
├──────────────┼────────────────────┼─────────────────────┤
│ Same Lane    │                    │                     │
│ < 1.0s TTC   │ CRITICAL           │ BRAKE IMMEDIATELY   │
│ < 1.5s TTC   │ WARNING            │ STRONG DECELERATE   │
│ < 2.5s TTC   │ CAUTION            │ DECELERATE/MONITOR  │
│ > 2.5s TTC   │ SAFE               │ MAINTAIN SPEED      │
├──────────────┼────────────────────┼─────────────────────┤
│ Left Lane    │ INFO               │ MONITOR LEFT LANE   │
│ Right Lane   │ INFO               │ MONITOR RIGHT LANE  │
└──────────────┴────────────────────┴─────────────────────┘
```

---

## Configuration & Thresholds

### Same Lane Thresholds (Collision Risk)

```python
TTC_CRITICAL = 1.0   seconds
TTC_WARNING  = 1.5   seconds  
TTC_SAFE     = 2.5   seconds

DRAC_CRITICAL = 3.35 m/s²  (Deceleration Rate to Avoid Collision)
DRAC_WARNING  = 2.0  m/s²

DISTANCE_CRITICAL = 10.0 meters
DISTANCE_WARNING  = 15.0 meters
```

### Adjacent Lane Thresholds (Informational Only)

```python
# Adjacent lanes: Vehicle approaching < 15m = "BE_AWARE"
# Adjacent lanes: Vehicle > 15m = "MONITOR"
# No collision warnings generated
```

---

## Troubleshooting

### Issue: Red warnings for vehicles clearly in other lanes

**Cause:** Vehicle bounding box centers detected as same-lane due to vehicle width

**Solution:** The system checks lane coverage percentage. Large vehicles spanning lanes get confidence-weighted assessment. If > 50% in center zone = same lane.

**Fix:** Bounding boxes are properly extracted by YOLO, lane detection is working.

---

### Issue: Orange warning for closely approaching left/right lane vehicle

**Expected Behavior:** Adjacent lane vehicles should show blue/orange INFO, not RED collision warnings

**Solution:** This is correct - adjacent lanes show approach speed but not collision warnings

---

## Testing the System

### Test Case 1: Same-Lane Vehicle (Approaching)

```bash
# Video with vehicle approaching from rear in same lane
./main.sh → Option 2 → Select video with same-lane approaching vehicle

Expected Output:
- RED bounding box
- TTC metrics displayed
- "Apply strong brakes immediately!" message
```

### Test Case 2: Adjacent-Lane Vehicle (Approaching)

```bash
# Video with vehicle approaching in left/right lane
./main.sh → Option 2 → Select video with adjacent-lane approaching vehicle

Expected Output:
- BLUE/ORANGE bounding box
- "LEFT LANE: Vehicle Xm away" message
- "Monitor vehicle in left lane" action
- NO collision warnings
```

### Test Case 3: Multiple Vehicles (Mixed Lanes)

```bash
# Video with vehicles in multiple lanes
./main.sh → Option 2 → Select multi-lane video

Expected Output:
- RED box for same-lane vehicles (collision risk)
- BLUE boxes for left-lane vehicles (awareness)
- ORANGE boxes for right-lane vehicles (awareness)
- Each with appropriate rider action
```

---

## Rider Actions Summary

### When to BRAKE (RED - CRITICAL)

- Same lane vehicle < 1.0 seconds away
- Closing gap rapidly (relative speed > 10 km/h)
- System says: "Apply strong brakes immediately!"

### When to DECELERATE (ORANGE - WARNING)

- Same lane vehicle 1.0-1.5 seconds away
- Vehicle approaching but slightly slower gap closure
- System says: "Reduce speed gradually" or "Decelerate aggressively!"

### When to MONITOR (YELLOW/BLUE - CAUTION/INFO)

- Same lane vehicle 1.5-2.5 seconds away
- OR any vehicle in adjacent lanes
- System says: "Monitor distance" or "Monitor left/right lane"

### When to MAINTAIN (GREEN - SAFE)

- Same lane vehicle > 2.5 seconds away
- All vehicles in adjacent lanes at safe distance
- System says: "Continue normal driving"

---

## Summary

**Key Feature:** Lane-aware safety that prevents false collision alarms from adjacent-lane traffic while providing critical warnings for same-lane collision risks.

**Result:** Rider gets actionable, context-specific guidance instead of generic alerts.

**Safety Benefit:** Distinguishes between imminent collision threats (same lane) and general awareness (adjacent lanes).
