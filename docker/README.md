# Direct Visual LiDAR Calibration Docker Setup

This folder contains Docker configurations for running the [direct_visual_lidar_calibration](https://github.com/koide3/direct_visual_lidar_calibration) toolbox.

## Available Docker Images

- `Dockerfile` - Base image with core calibration functionality
- `Dockerfile_with_sift` - Enhanced image with SIFT for automatic feature matching (free for academic and commercial use)
- `Dockerfile_with_superglue` - Image with SuperGlue (not recommended for commercial use, just for feasibility)

## Prerequisites

- Docker installed and running
- ROS bag files with synchronized camera and LiDAR data
- Sufficient disk space for Docker images and data processing

## Quick Start

### 1. Build the Docker Image

For the SIFT-enabled version (recommended for commercial use):

```bash
cd docker/humble
docker build --platform linux/amd64 -f Dockerfile_with_sift -t cal-w-sift .
```

For the base version:

```bash
cd docker/humble
docker build --platform linux/amd64 -f Dockerfile -t cal-base .
```

### 2. Prepare Your Data

Set up your data paths (adjust these to your actual locations):

```bash
# Set paths to your data
preprocessed_path="/path/to/your/preprocessed/output"
input_bags_path="/path/to/your/rosbag/files"

# Create output directory if it doesn't exist
mkdir -p "$preprocessed_path"
```

### 3. Preprocessing

Run preprocessing to extract and prepare camera images and LiDAR point clouds:

```bash
docker run \
  --rm \
  -v $input_bags_path:/tmp/input_bags \
  -v $preprocessed_path:/tmp/preprocessed \
  cal-w-sift \
  ros2 run direct_visual_lidar_calibration preprocess /tmp/input_bags /tmp/preprocessed \
    --image_topic /ipcamera/image_raw/compressed \
    --points_topic /livox/lidar \
    --camera_model plumb_bob \
    --camera_intrinsics 1309,1740,728,544 \
    --camera_distortion_coeffs 0,0,0,0,0
```

**Note:** Adjust the following parameters for your setup:
- `--image_topic`: Your camera topic name
- `--points_topic`: Your LiDAR topic name  
- `--camera_intrinsics`: Your camera intrinsic parameters (fx,fy,cx,cy)
- `--camera_distortion_coeffs`: Your camera distortion coefficients

### 4. Feature Matching (SIFT Method)

Run automatic feature matching using SIFT:

```bash
# Find matches using SIFT
docker run \
  --rm \
  -v $preprocessed_path:/tmp/preprocessed \
  cal-w-sift \
  ros2 run direct_visual_lidar_calibration find_matches_sift.py /tmp/preprocessed \
    --max_keypoints 8000 \
    --keypoint_threshold 0.02 \
    --match_ratio 0.8 \
    --min_matches 15

# Generate initial guess from matches
docker run \
  --rm \
  -v $preprocessed_path:/tmp/preprocessed \
  cal-w-sift \
  ros2 run direct_visual_lidar_calibration initial_guess_auto /tmp/preprocessed
```

### 5. Final Calibration

Run the calibration optimization:

```bash
docker run \
  --rm \
  -v $preprocessed_path:/tmp/preprocessed \
  cal-w-sift \
  bash -c "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & export DISPLAY=:99; ros2 run direct_visual_lidar_calibration calibrate /tmp/preprocessed --background --auto_quit"
```

## Example with Sample Data

Here's a complete example using sample data:

```bash
# Example paths (adjust to your setup)
preprocessed_path="/Users/jhathaway/data/abf_bags/motion_recording_2025_06_10_16_02_51/preprocessed"
input_bags_path="/Users/jhathaway/data/abf_bags/motion_recording_2025_06_10_16_02_51/chunk_001"

# Create output directory
mkdir -p "$preprocessed_path"

# Step 1: Preprocess
docker run \
  --rm \
  -v $input_bags_path:/tmp/input_bags \
  -v $preprocessed_path:/tmp/preprocessed \
  cal-w-sift \
  ros2 run direct_visual_lidar_calibration preprocess /tmp/input_bags /tmp/preprocessed \
    --image_topic /ipcamera/image_raw/compressed \
    --points_topic /livox/lidar \
    --camera_model plumb_bob \
    --camera_intrinsics 1309,1740,728,544 \
    --camera_distortion_coeffs 0,0,0,0,0

# Step 2: Find matches with SIFT
docker run \
  --rm \
  -v $preprocessed_path:/tmp/preprocessed \
  cal-w-sift \
  ros2 run direct_visual_lidar_calibration find_matches_sift.py /tmp/preprocessed \
    --max_keypoints 8000 \
    --keypoint_threshold 0.02 \
    --edge_threshold 20.0 \
    --match_ratio 0.8 \
    --min_matches 15

# Step 3: Generate initial guess
docker run \
  --rm \
  -v $preprocessed_path:/tmp/preprocessed \
  cal-w-sift \
  ros2 run direct_visual_lidar_calibration initial_guess_auto /tmp/preprocessed

# Step 4: Final calibration
docker run \
  --rm \
  -v $preprocessed_path:/tmp/preprocessed \
  cal-w-sift \
  bash -c "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & export DISPLAY=:99; ros2 run direct_visual_lidar_calibration calibrate /tmp/preprocessed --background --auto_quit"
```

## Advanced SIFT Parameters

To fine-tune SIFT performance, you can modify these parameters:

```bash
# SIFT matching with custom parameters
docker run \
  --rm \
  -v $preprocessed_path:/tmp/preprocessed \
  cal-w-sift \
  ros2 run direct_visual_lidar_calibration find_matches_sift.py /tmp/preprocessed \
    --max_keypoints 10000 \              # Maximum number of keypoints
    --keypoint_threshold 0.01 \          # Contrast threshold (lower = more features)
    --edge_threshold 15.0 \              # Edge threshold (higher = more edge features)
    --sigma 1.6 \                        # Gaussian sigma
    --match_ratio 0.75 \                 # Lowe's ratio test threshold
    --min_matches 20 \                   # Minimum number of matches required
    --show_keypoints \                   # Show detected keypoints
    --rotate_camera 0 \                  # Rotate camera image (0, 90, 180, 270 degrees)
    --rotate_lidar 0                     # Rotate LiDAR image (0, 90, 180, 270 degrees)
```

## Output

After successful calibration, you'll find the results in your `preprocessed_path` directory, including:
- Calibration transformation matrix
- Visualization images
- Detailed calibration report

## SIFT vs SuperGlue Advantages

### SIFT
- ✅ **Free for commercial use** - No license restrictions
- ✅ **Robust and proven** - Established algorithm with decades of development
- ✅ **Lower computational requirements** - No GPU required
- ✅ **Fast** - Efficient processing
- ✅ **Tunable parameters** - Fine control over feature detection

### SuperGlue
- ⚠️ **License restrictions** - Not recommended for commercial use
- ✅ **Higher accuracy** - More modern deep learning algorithm
- ❌ **Requires GPU** - Higher resource consumption
- ❌ **Less control** - Limited parameters for tuning

## Troubleshooting

### Common Issues

1. **Permission errors**: Make sure Docker has access to your data directories
2. **Platform issues on Mac**: Use `--platform linux/amd64` flag when building
3. **Memory issues**: Ensure sufficient RAM (8GB+ recommended) for large datasets
4. **Display issues**: The Xvfb setup handles headless operation automatically

### Few SIFT Matches

If SIFT finds few matches, try:

```bash
# More features, lower threshold
--max_keypoints 15000 --keypoint_threshold 0.005

# More permissive match ratio
--match_ratio 0.85

# Lower minimum matches
--min_matches 10
```

### Too Many False Matches

If there are too many incorrect matches:

```bash
# Stricter threshold for better features
--keypoint_threshold 0.03

# Stricter match ratio
--match_ratio 0.7

# More minimum matches for better filtering
--min_matches 25
```

### Getting Camera Intrinsics

For better calibration results, obtain accurate camera intrinsics using:
- Camera manufacturer specifications
- Camera calibration tools (e.g., OpenCV calibration)
- ROS camera calibration package

## References

- [Original Repository](https://github.com/koide3/direct_visual_lidar_calibration)
- [Documentation](https://koide3.github.io/direct_visual_lidar_calibration/)
- [Paper](https://arxiv.org/abs/2302.05094): General, Single-shot, Target-less, and Automatic LiDAR-Camera Extrinsic Calibration Toolbox
- [Original SIFT Paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf): Distinctive Image Features from Scale-Invariant Keypoints 