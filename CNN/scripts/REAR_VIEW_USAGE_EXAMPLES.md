# Rear-View ADAS - Usage Examples & Test Cases

## Example 1: Basic Processing with Safety Assessment

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from inference.video_inference import VideoDetector, DetectionLogger, process_video

# Process a video file
input_video = "test_video.mp4"
output_video = "result_with_safety.mp4"

success = process_video(
    input_path=input_video,
    output_path=output_video,
    device='cuda',  # Use GPU for speed
)

if success:
    print(f"✅ Video processed successfully!")
    print(f"   Output: {output_video}")
    print(f"   CSV Log: {Path(output_video).stem}_detections.csv")
```

## Example 2: Analyzing Safety Events from CSV

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load detection log
df = pd.read_csv('result_detections.csv')

# Parse numeric columns
df['ttc_s'] = pd.to_numeric(df['ttc_s'], errors='coerce')
df['distance_m'] = pd.to_numeric(df['distance_m'], errors='coerce')
df['drac_ms2'] = pd.to_numeric(df['drac_ms2'], errors='coerce')

print("=" * 60)
print("SAFETY ANALYSIS SUMMARY")
print("=" * 60)

# Overall statistics
print(f"\nTotal frames processed: {len(df)}")
print(f"Unique vehicles tracked: {df['track_id'].nunique()}")

# Safety level distribution
print("\n--- Safety Level Distribution ---")
safety_dist = df['safety_level'].value_counts()
for level, count in safety_dist.items():
    percentage = (count / len(df)) * 100
    print(f"{level:12s}: {count:6d} ({percentage:5.1f}%)")

# Critical events
critical_events = df[df['safety_level'] == 'CRITICAL']
if len(critical_events) > 0:
    print(f"\n⚠️ CRITICAL EVENTS DETECTED: {len(critical_events)}")
    print("\nCritical Event Timeline:")
    for idx, row in critical_events.iterrows():
        print(f"  Frame {row['frame_number']:5d} @ {row['timestamp_s']:.2f}s: "
              f"{row['vehicle_class']} at {row['distance_m']:.1f}m "
              f"(TTC: {row['ttc_s']:.2f}s, DRAC: {row['drac_ms2']:.2f}m/s²)")

# Distance statistics
df_valid = df[df['distance_m'] > 0]
print("\n--- Distance Statistics ---")
print(f"Min Distance: {df_valid['distance_m'].min():.2f}m")
print(f"Max Distance: {df_valid['distance_m'].max():.2f}m")
print(f"Mean Distance: {df_valid['distance_m'].mean():.2f}m")
print(f"Median Distance: {df_valid['distance_m'].median():.2f}m")

# TTC statistics for approaching vehicles
approaching = df[(df['motion_state'] == 'approaching') & (df['ttc_s'] > 0)]
if len(approaching) > 0:
    print("\n--- TTC Statistics (Approaching Vehicles) ---")
    print(f"Min TTC: {approaching['ttc_s'].min():.2f}s")
    print(f"Max TTC: {approaching['ttc_s'].max():.2f}s")
    print(f"Mean TTC: {approaching['ttc_s'].mean():.2f}s")
    print(f"Below 1.0s: {len(approaching[approaching['ttc_s'] < 1.0])} events")
    print(f"Below 1.5s: {len(approaching[approaching['ttc_s'] < 1.5])} events")

# Vehicle class analysis
print("\n--- Detections by Vehicle Class ---")
class_dist = df['vehicle_class'].value_counts()
for vclass, count in class_dist.items():
    avg_dist = df[df['vehicle_class'] == vclass]['distance_m'].mean()
    avg_ttc = df[df['vehicle_class'] == vclass]['ttc_s'].mean()
    print(f"{vclass:20s}: {count:5d} detections, "
          f"Avg dist: {avg_dist:6.1f}m, Avg TTC: {avg_ttc:5.2f}s")

print("\n" + "=" * 60)
```

## Example 3: Visualizing Safety Metrics Over Time

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('result_detections.csv')

# Convert to numeric
df['timestamp_s'] = pd.to_numeric(df['timestamp_s'], errors='coerce')
df['distance_m'] = pd.to_numeric(df['distance_m'], errors='coerce')
df['ttc_s'] = pd.to_numeric(df['ttc_s'], errors='coerce')
df['speed_kmh'] = pd.to_numeric(df['speed_kmh'], errors='coerce')

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Distance over time
ax1 = axes[0]
valid_dist = df[df['distance_m'] > 0]
ax1.plot(valid_dist['timestamp_s'], valid_dist['distance_m'], 'b-', alpha=0.7)
ax1.axhline(y=5.0, color='r', linestyle='--', label='Critical (5m)', linewidth=2)
ax1.axhline(y=10.0, color='orange', linestyle='--', label='Warning (10m)', linewidth=2)
ax1.axhline(y=15.0, color='yellow', linestyle='--', label='Caution (15m)', linewidth=2)
ax1.set_ylabel('Distance (meters)', fontsize=11)
ax1.set_title('Distance from Rear Vehicle Over Time', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: TTC and DRAC
ax2 = axes[1]
valid_ttc = df[(df['ttc_s'] > 0) & (df['ttc_s'] < 10)]
ax2.plot(valid_ttc['timestamp_s'], valid_ttc['ttc_s'], 'g-', label='TTC', alpha=0.7, linewidth=2)
ax2.axhline(y=1.0, color='r', linestyle='--', label='Critical TTC (1.0s)', linewidth=2)
ax2.axhline(y=1.5, color='orange', linestyle='--', label='Warning TTC (1.5s)', linewidth=2)
ax2.set_ylabel('TTC (seconds)', fontsize=11)
ax2.set_title('Time to Collision (TTC) Over Time', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Speed and Safety Level
ax3 = axes[2]
speed_data = df[df['speed_kmh'] > 0]
colors = {'CRITICAL': 'red', 'WARNING': 'orange', 'CAUTION': 'yellow', 'SAFE': 'green', 'unknown': 'gray'}
color_list = [colors.get(level, 'gray') for level in df['safety_level']]

scatter = ax3.scatter(df['timestamp_s'], df['speed_kmh'], c=color_list, s=50, alpha=0.6)
ax3.set_ylabel('Speed (km/h)', fontsize=11)
ax3.set_xlabel('Time (seconds)', fontsize=11)
ax3.set_title('Vehicle Speed with Safety Level', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=level) 
                   for level, color in colors.items()]
ax3.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('safety_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: safety_analysis.png")
plt.show()
```

## Example 4: Real-time Monitoring (Single Detection)

```python
from inference.video_inference import RearViewSafetyAssessment

# Create safety assessment engine
safety = RearViewSafetyAssessment()

# Simulated detection data
test_cases = [
    {
        'distance_m': 25.0,
        'ego_speed_kmh': 80.0,
        'rear_speed_kmh': 80.0,
        'description': 'Matching speed, safe'
    },
    {
        'distance_m': 20.0,
        'ego_speed_kmh': 80.0,
        'rear_speed_kmh': 100.0,
        'description': 'Approaching, moderate risk'
    },
    {
        'distance_m': 10.0,
        'ego_speed_kmh': 60.0,
        'rear_speed_kmh': 120.0,
        'description': 'Rapid approach, HIGH RISK'
    },
    {
        'distance_m': 5.0,
        'ego_speed_kmh': 50.0,
        'rear_speed_kmh': 110.0,
        'description': 'Critical distance, IMMINENT'
    },
]

print("\n" + "=" * 80)
print("SAFETY ASSESSMENT TEST CASES")
print("=" * 80)

for i, test in enumerate(test_cases, 1):
    ego_speed_ms = test['ego_speed_kmh'] / 3.6
    rear_speed_ms = test['rear_speed_kmh'] / 3.6
    
    ttc = safety.calculate_ttc(test['distance_m'], ego_speed_ms, rear_speed_ms)
    mttc = safety.calculate_mttc(test['distance_m'], ego_speed_ms, rear_speed_ms, 0.0, 0.0)
    pet = safety.calculate_pet(test['distance_m'], ego_speed_ms, rear_speed_ms)
    drac = safety.calculate_drac(test['distance_m'], ego_speed_ms, rear_speed_ms)
    
    result = safety.assess_risk_level(ttc, mttc, pet, drac, 
                                      test['distance_m'], ego_speed_ms, rear_speed_ms)
    
    print(f"\nTest Case {i}: {test['description']}")
    print(f"  Configuration:")
    print(f"    Distance: {test['distance_m']:.1f}m")
    print(f"    Ego Speed: {test['ego_speed_kmh']:.1f} km/h")
    print(f"    Rear Speed: {test['rear_speed_kmh']:.1f} km/h")
    print(f"  Metrics:")
    print(f"    TTC: {ttc:.2f}s" if ttc is not None else "    TTC: N/A")
    print(f"    MTTC: {mttc:.2f}s" if mttc is not None else "    MTTC: N/A")
    print(f"    PET: {pet:.2f}s")
    print(f"    DRAC: {drac:.2f}m/s²" if drac != float('inf') else "    DRAC: ∞")
    print(f"  Assessment:")
    print(f"    Level: {result['level']}")
    print(f"    Alert: {result['alert_type']}")
    print(f"    Message: {result['message']}")
    print(f"    Confidence: {result['confidence']:.1%}")

print("\n" + "=" * 80)
```

## Example 5: Batch Processing Multiple Videos

```python
from pathlib import Path
import csv
from inference.video_inference import process_video

# Process multiple videos
video_dir = Path("test_videos")
videos = list(video_dir.glob("*.mp4"))

results = []

for video_path in videos:
    output_path = video_dir / "results" / f"{video_path.stem}_result.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎬 Processing: {video_path.name}")
    
    try:
        success = process_video(
            str(video_path),
            str(output_path),
            device='cuda'
        )
        
        if success:
            # Analyze the safety log
            csv_log = output_path.parent / f"{output_path.stem}_detections.csv"
            
            import pandas as pd
            df = pd.read_csv(csv_log)
            critical_count = len(df[df['safety_level'] == 'CRITICAL'])
            warning_count = len(df[df['safety_level'] == 'WARNING'])
            
            results.append({
                'video': video_path.name,
                'status': 'SUCCESS',
                'output': str(output_path),
                'critical_events': critical_count,
                'warning_events': warning_count,
            })
            
            print(f"✅ Processed: {video_path.name}")
            print(f"   - Critical events: {critical_count}")
            print(f"   - Warning events: {warning_count}")
            
    except Exception as e:
        results.append({
            'video': video_path.name,
            'status': 'FAILED',
            'error': str(e),
        })
        print(f"❌ Failed: {video_path.name} - {e}")

# Save batch results
with open('batch_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
    writer.writeheader()
    writer.writerows(results)

print(f"\n📊 Batch processing complete! Results saved to: batch_results.csv")
```

## Example 6: Unit Testing Safety Calculations

```python
import pytest
from inference.video_inference import RearViewSafetyAssessment

class TestRearViewSafety:
    """Unit tests for safety assessment calculations"""
    
    @pytest.fixture
    def safety(self):
        return RearViewSafetyAssessment()
    
    def test_ttc_no_threat(self, safety):
        """Same speed vehicles - no collision"""
        ttc = safety.calculate_ttc(
            distance_m=20.0,
            ego_speed_ms=10.0,
            rear_speed_ms=10.0  # Same speed
        )
        assert ttc is None  # No collision
    
    def test_ttc_critical(self, safety):
        """Close vehicle approaching rapidly"""
        ttc = safety.calculate_ttc(
            distance_m=10.0,
            ego_speed_ms=10.0,
            rear_speed_ms=20.0  # 10 m/s faster
        )
        assert ttc is not None
        assert ttc < 1.5  # Should be critical
    
    def test_drac_calculation(self, safety):
        """Test deceleration rate calculation"""
        drac = safety.calculate_drac(
            distance_m=20.0,
            ego_speed_ms=13.89,  # 50 km/h
            rear_speed_ms=27.78,  # 100 km/h
            reaction_time=1.0
        )
        assert drac > 0
        assert drac < 20  # Reasonable deceleration
    
    def test_risk_assessment_critical(self, safety):
        """Test critical risk assessment"""
        result = safety.assess_risk_level(
            ttc_s=0.8,
            mttc_s=None,
            pet_s=0.5,
            drac_ms2=5.0,
            distance_m=5.0,
            ego_speed_ms=10.0,
            rear_speed_ms=20.0
        )
        assert result['level'] == 'CRITICAL'
        assert result['alert_type'] in ['collision_imminent', 'collision_warning']
    
    def test_risk_assessment_safe(self, safety):
        """Test safe scenario"""
        result = safety.assess_risk_level(
            ttc_s=5.0,
            mttc_s=5.0,
            pet_s=6.0,
            drac_ms2=0.5,
            distance_m=30.0,
            ego_speed_ms=10.0,
            rear_speed_ms=10.0
        )
        assert result['level'] == 'SAFE'

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

## Example 7: Integration Test - Full Pipeline

```python
import cv2
import numpy as np
from pathlib import Path
from inference.video_inference import VideoDetector, RearSideUseCaseValidator

# Create a test frame (1920x1080)
frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

# Initialize detector
detector = VideoDetector(device='cuda', fps=30)

# Simulate a detection
test_detection = {
    'bbox': [800, 400, 1100, 650],  # [x1, y1, x2, y2]
    'class': 'Sedan',
    'confidence': 0.95,
    'distance': 15.0,  # 15 meters away
    'speed': 25.5,
    'motion': 'approaching',
    'track_id': 1,
    'distance_metadata': {
        'classical_fused': 15.0,
        'ml': 14.8,
    }
}

# Test 1: Validate rear detection
validator = RearSideUseCaseValidator(1920, 1080)
validation = validator.is_valid_rear_detection(
    test_detection['bbox'],
    test_detection['bbox'][3] - test_detection['bbox'][1],
    test_detection['distance']
)

print("Rear Detection Validation:")
print(f"  Valid: {validation['is_valid']}")
print(f"  Confidence: {validation['confidence']:.2%}")
print(f"  Issues: {[i for i in validation['issues'] if i]}")

# Test 2: Safety assessment
safety_result = detector.calculate_rear_safety_assessment(test_detection)
print(f"\nSafety Assessment:")
print(f"  Level: {safety_result['level']}")
print(f"  Alert: {safety_result['alert_type']}")
print(f"  TTC: {safety_result['ttc']:.2f}s" if safety_result['ttc'] else "  TTC: N/A")
print(f"  DRAC: {safety_result['drac']:.2f}m/s²" if safety_result['drac'] else "  DRAC: N/A")

# Test 3: Scenario validation
scenario = validator.validate_rear_scenario([test_detection], ego_speed_ms=0)
print(f"\nScenario Validation:")
print(f"  Type: {scenario['scenario_type']}")
print(f"  Threat: {scenario['threat_level']}")
print(f"  Critical vehicles: {scenario['critical_vehicles_count']}")

# Test 4: Visualization
detections = [test_detection]
safety_assessments, scene = detector.validate_and_assess_rear_scenario(detections)
annotated = detector.draw_detections(frame, detections)

# Save visualization
output_path = Path("test_visualization.png")
cv2.imwrite(str(output_path), annotated)
print(f"\n✅ Saved visualization: {output_path}")
```

## Running These Examples

```bash
# Run safety analysis
python -c "
import pandas as pd
df = pd.read_csv('result_detections.csv')
print(f'Total frames: {len(df)}')
print(f'Critical events: {len(df[df[\"safety_level\"] == \"CRITICAL\"])}')
"

# Run with specific video
python video_inference.py --input sample.mp4 --output sample_result.mp4

# Batch analysis
for video in *.mp4; do
    python video_inference.py --input "$video" --output "result_$video"
done
```

## Expected Output Examples

### Console Output:
```
🔥 Device: cuda
📦 Loading YOLO...
✅ YOLO loaded
📦 Loading Classifier: /path/to/classifier
✅ Classifier loaded
📹 Opening: test_video.mp4
✅ Video: 1920x1080 @ 30.00 FPS, 150 frames
💾 Saving to: result_video.mp4
📊 Logging to: result_video_detections.csv

🚀 Processing...

   Frame 150/150 (100.0%) - 25.3 FPS

✅ Processing complete!
   Frames: 150
   Time: 5.9s
   Avg FPS: 25.3
   Video Size: 45.23 MB

   CSV Detections: result_video_detections.csv
   CSV Size: 125.45 KB
   Total Detections Logged: 150 frames
```

### CSV Sample Row:
```
150,5,Sedan,0.92,8.5,22.3,approaching,850,300,1100,550,250,250,8.5,8.2,5.0,WARNING,collision_warning,1.35,1.32,2.50,2.85,0.87,approaching_vehicle
```

---

**Note**: These examples assume the video_inference.py module is properly installed and configured. Adjust paths and parameters as needed for your setup.
