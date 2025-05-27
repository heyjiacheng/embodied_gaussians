# Real-Time Object Tracking with Embodied Gaussians

This document explains how to use the real-time tracking system to track reconstructed objects using your own cameras.

## Prerequisites

1. You have successfully built an object using `build_simple_body.py`
2. Your cameras are set up and working with the `MultiRealsense` interface
3. You have camera extrinsics calibrated

## Basic Usage

The tracking script is located at `scripts/real_time_tracking.py`. Here's how to use it:

### Basic Command

```bash
python scripts/real_time_tracking.py objects/tblock.json \
    --extrinsics my_env/cameras_tf.json \
    --ground scripts/example_ground_plane.json \
    --visualize
```

### Command Line Arguments

- `body_json` (required): Path to the JSON file containing the body to track
- `--extrinsics`: Path to camera extrinsics file
- `--ground`: Path to ground plane definition (optional)
- `--fps`: Camera capture FPS (default: 30)
- `--tracking_fps`: Tracking update FPS (default: 60)
- `--visualize`: Show 3D visualization (default: True)
- `--save_tracking_data`: Save tracking data to file
- `--output_dir`: Directory to save tracking data
- `--convert_bgr_to_rgb`: Fix red/blue color swap (default: False)

### Example Commands

**Basic tracking with visualization:**
```bash
python scripts/real_time_tracking.py objects/tblock.json \
    --extrinsics my_env/cameras_tf.json \
    --visualize
```

**Track and save data without visualization:**
```bash
python scripts/real_time_tracking.py objects/tblock.json \
    --extrinsics my_env/cameras_tf.json \
    --save_tracking_data \
    --output_dir tracking_results \
    --no-visualize
```

**High-frequency tracking:**
```bash
python scripts/real_time_tracking.py objects/tblock.json \
    --extrinsics my_env/cameras_tf.json \
    --fps 60 \
    --tracking_fps 120 \
    --visualize
```

**Fix color channel issues (if red/blue are swapped):**
```bash
python scripts/real_time_tracking.py objects/tblock.json \
    --extrinsics my_env/cameras_tf.json \
    --convert_bgr_to_rgb \
    --visualize
```

## How It Works

### Tracking Process

1. **Initialization**: 
   - Loads the reconstructed object (gaussians + particles)
   - Sets up cameras with your extrinsics
   - Creates a physics environment

2. **Real-time Loop**:
   - Captures images from all cameras
   - Uses visual forces to align gaussian particles with observed imagery
   - Updates physics simulation
   - Records tracking data (if enabled)

3. **Visual Forces**:
   - Compares rendered gaussians with camera images
   - Computes gradients to minimize photometric error
   - Applies forces to move gaussians to match observations

### Data Output

When `--save_tracking_data` is enabled, the system saves:

- **Gaussian positions**: 3D positions of all gaussian particles over time
- **Gaussian colors**: RGB colors of gaussians (may adapt during tracking)
- **Gaussian opacities**: Transparency values
- **Body transforms**: Physics body poses
- **Timestamps**: Precise timing information

The output is saved as JSON in the format:
```json
{
  "body_name": "tblock",
  "start_time": 1643723400.0,
  "duration": 120.5,
  "num_frames": 7230,
  "camera_serials": ["123456789", "987654321"],
  "tracking_data": [
    {
      "timestamp": 0.0,
      "gaussian_means": [[x1, y1, z1], [x2, y2, z2], ...],
      "gaussian_colors": [[r1, g1, b1], [r2, g2, b2], ...],
      "gaussian_opacities": [o1, o2, ...],
      "body_transforms": [...]
    },
    ...
  ]
}
```

## GUI Controls

When visualization is enabled, you'll see several information panels:

### Tracking Stats
- Shows elapsed time and recorded frames
- **Reset Tracking**: Resets object to initial pose
- **Save Results**: Saves current tracking data

### Gaussian Particles
- Total number of gaussians
- Mean position of all particles
- Individual gaussian positions (first 5)
- Average opacity

### Camera Info
- Camera count and serial numbers
- Camera intrinsics (fx, fy, cx, cy)
- Visual tracking status
- Frame dimensions

## Tips for Good Tracking

1. **Lighting**: Ensure consistent, good lighting conditions
2. **Camera Positioning**: Position cameras to cover the object from multiple angles
3. **Initial Placement**: Start with the object in approximately the same pose as during reconstruction
4. **Movement**: Move the object slowly for better tracking accuracy
5. **Occlusion**: Avoid complete occlusion by too many fingers/hands

## Troubleshooting

### Camera Issues
- Check that cameras are detected: `realsense-viewer`
- Verify camera serials match extrinsics file
- Ensure cameras have depth enabled

### Tracking Problems
- Object jumps around: Reduce tracking_fps or improve lighting
- Poor visual tracking: Check that object appearance matches reconstruction
- Performance issues: Lower fps or disable depth if not needed

### Visualization Issues
- **Static camera images**: This has been fixed in the latest version. Camera images should now update in real-time
- Camera view not updating: Check that "Frames initialized: True" appears in the Camera Info panel
- Black camera views: Ensure proper lighting and camera exposure settings
- **Red/blue colors swapped**: Add `--convert_bgr_to_rgb` flag to fix color channel order

### Common Error Messages
- "Camera X is not known": Camera serial not in extrinsics file
- "No color frame available": Camera might not support color stream
- "Frame timeout": Camera connection issue or insufficient lighting
- "Frames not initialized": Camera setup failed, check camera connections

## Testing Camera Updates

To verify that camera image updates work correctly, you can run the animation test:

```bash
python scripts/test_camera_updates.py
```

This will show animated test patterns that should move smoothly in the visualization. If the patterns are static, there may be an issue with the frame update system.

## Advanced Usage

### Custom Segmentation
You can modify the `_capture_camera_data` method to add segmentation:

```python
# Replace full image mask with actual segmentation
mask = np.ones((color.shape[0], color.shape[1]), dtype=bool)
# with:
mask = your_segmentation_function(color_rgb)
```

### Multiple Objects
To track multiple objects, create separate tracking instances and run them in parallel.

### Integration with Other Systems
The tracking data can be used for:
- Robot control systems
- AR/VR applications  
- Data collection for machine learning
- Real-time physics simulation 