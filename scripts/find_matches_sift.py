#!/usr/bin/env python3
# SIFT-based correspondence matching alternative to SuperGlue
import sys
import cv2
import math
import json
import numpy
import argparse
import matplotlib

def main():
  print('\033[92m' + '*******************************************************************************************' + '\033[0m')
  print('\033[92m' + '* Using SIFT for correspondence matching - Free for academic and commercial use         *' + '\033[0m')
  print('\033[92m' + '*******************************************************************************************' + '\033[0m')

  parser = argparse.ArgumentParser(description='Initial guess estimation based on SIFT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('data_path', help='Input data path')
  parser.add_argument('--max_keypoints', type=int, default=8000, help='Maximum number of keypoints detected by SIFT')
  parser.add_argument('--keypoint_threshold', type=float, default=0.02, help='SIFT contrast threshold for filtering weak features (lower = more features)')
  parser.add_argument('--edge_threshold', type=float, default=20.0, help='SIFT edge threshold for filtering edge-like features (higher = more edge features)')
  parser.add_argument('--sigma', type=float, default=1.6, help='SIFT sigma of the Gaussian applied to the input image at the octave #0')
  parser.add_argument('--match_ratio', type=float, default=0.8, help='Lowe\'s ratio test threshold for good matches (higher = more matches)')
  parser.add_argument('--min_matches', type=int, default=15, help='Minimum number of matches required')
  parser.add_argument('--show_keypoints', action='store_true', help='Show the detected keypoints')
  parser.add_argument('--rotate_camera', type=int, default=0, help='Rotate camera image before matching (CW 0, 90, 180, or 270) (CW)')
  parser.add_argument('--rotate_lidar', type=int, default=0, help='Rotate LiDAR image before matching (0, 90, 180, or 270) (CW)')

  opt = parser.parse_args()
  print(opt)

  def angle_to_rot(angle, image_shape):
    width, height = image_shape[:2]

    if angle == 90:
      code = cv2.ROTATE_90_CLOCKWISE
      func = lambda x: numpy.stack([x[:, 1], width - x[:, 0]], axis=1)
    elif angle == 180:
      code = cv2.ROTATE_180
      func = lambda x: numpy.stack([height - x[:, 0], width - x[:, 1]], axis=1)
    elif angle == 270:
      code = cv2.ROTATE_90_COUNTERCLOCKWISE
      func = lambda x: numpy.stack([height - x[:, 1], x[:, 0]], axis=1)
    else:
      print('error: unsupported rotation angle %d' % angle)
      exit(1)

    return code, func

  data_path = opt.data_path
  with open(data_path + '/calib.json', 'r') as f:
    calib_config = json.load(f)

  # Initialize SIFT detector with parameters tuned for LiDAR-camera matching
  sift = cv2.SIFT_create(
    nfeatures=opt.max_keypoints,
    contrastThreshold=opt.keypoint_threshold,
    edgeThreshold=opt.edge_threshold,
    sigma=opt.sigma
  )
  
  def preprocess_image(img, is_lidar=False):
    """Preprocess images to improve feature matching between camera and LiDAR"""
    if is_lidar:
      # LiDAR intensity preprocessing
      # Apply histogram equalization to improve contrast
      img = cv2.equalizeHist(img)
      # Apply bilateral filter to reduce noise while preserving edges
      img = cv2.bilateralFilter(img, 9, 75, 75)
    else:
      # Camera image preprocessing
      # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
      img = clahe.apply(img)
    
    return img

  # Initialize matcher with FLANN
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)

  for bag_name in calib_config['meta']['bag_names']:
    print('processing %s' % bag_name)

    camera_image = cv2.imread('%s/%s.png' % (data_path, bag_name), 0)
    lidar_image = cv2.imread('%s/%s_lidar_intensities.png' % (data_path, bag_name), 0)

    if opt.rotate_camera:
      code, camera_R_inv = angle_to_rot(opt.rotate_camera, camera_image.shape)
      camera_image = cv2.rotate(camera_image, code)
    if opt.rotate_lidar:
      code, lidar_R_inv = angle_to_rot(opt.rotate_lidar, lidar_image.shape)
      lidar_image = cv2.rotate(lidar_image, code)

    # Preprocess images to improve matching
    camera_processed = preprocess_image(camera_image, is_lidar=False)
    lidar_processed = preprocess_image(lidar_image, is_lidar=True)

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(camera_processed, None)
    kp2, des2 = sift.detectAndCompute(lidar_processed, None)

    if des1 is None or des2 is None or len(des1) < opt.min_matches or len(des2) < opt.min_matches:
      print(f'Warning: Not enough keypoints found in {bag_name} (camera: {len(des1) if des1 is not None else 0}, lidar: {len(des2) if des2 is not None else 0})')
      # Create empty results
      result = {'kpts0': [], 'kpts1': [], 'matches': [], 'confidence': []}
      with open('%s/%s_matches.json' % (data_path, bag_name), 'w') as f:
        json.dump(result, f)
      continue

    # Match descriptors using FLANN
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
      if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < opt.match_ratio * n.distance:
          good_matches.append(m)

    # Additional filtering: Remove outliers using homography if enough matches
    if len(good_matches) >= 10:
      src_pts = numpy.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      dst_pts = numpy.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      
      try:
        # Use RANSAC to find homography and filter outliers
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
          good_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
          print(f'Filtered to {len(good_matches)} matches after homography RANSAC')
      except:
        print('Homography filtering failed, using all matches')

    print(f'Found {len(good_matches)} good matches for {bag_name}')

    if len(good_matches) < opt.min_matches:
      print(f'Warning: Not enough good matches found in {bag_name} ({len(good_matches)} < {opt.min_matches})')

    # Extract matched keypoints
    kpts0 = numpy.array([kp1[m.queryIdx].pt for m in good_matches], dtype=numpy.float32)
    kpts1 = numpy.array([kp2[m.trainIdx].pt for m in good_matches], dtype=numpy.float32)
    
    # Create matches array (index mapping)
    matches_array = numpy.arange(len(good_matches), dtype=numpy.int32)
    
    # Create confidence scores (inverse of distance, normalized)
    distances = numpy.array([m.distance for m in good_matches])
    if len(distances) > 0:
      confidence = 1.0 / (1.0 + distances)
      confidence = confidence / numpy.max(confidence)
    else:
      confidence = numpy.array([])

    kpts0_ = kpts0
    kpts1_ = kpts1

    if opt.rotate_camera and len(kpts0_) > 0:
      kpts0_ = camera_R_inv(kpts0_)
    if opt.rotate_lidar and len(kpts1_) > 0:
      kpts1_ = lidar_R_inv(kpts1_)

    result = { 
      'kpts0': kpts0_.flatten().tolist() if len(kpts0_) > 0 else [], 
      'kpts1': kpts1_.flatten().tolist() if len(kpts1_) > 0 else [], 
      'matches': matches_array.flatten().tolist() if len(matches_array) > 0 else [], 
      'confidence': confidence.flatten().tolist() if len(confidence) > 0 else []
    }
    with open('%s/%s_matches.json' % (data_path, bag_name), 'w') as f:
      json.dump(result, f)

    # Visualization
    if len(good_matches) > 0:
      camera_canvas = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.cvtColor(lidar_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.resize(lidar_canvas, (camera_image.shape[1], camera_image.shape[0]))

      sx = camera_image.shape[1] / lidar_image.shape[1]
      sy = camera_image.shape[0] / lidar_image.shape[0]

      # Adjust lidar keypoints for visualization
      kpts1_vis = kpts1.copy()
      kpts1_vis[:, 0] = kpts1_vis[:, 0] * sx + camera_image.shape[1]
      kpts1_vis[:, 1] = kpts1_vis[:, 1] * sy

      canvas = numpy.concatenate([camera_canvas, lidar_canvas], axis=1)
      
      # Draw keypoints
      for kp in kpts0:
        cv2.circle(canvas, (int(kp[0]), int(kp[1])), 3, (255, 255, 255))
      for kp in kpts1_vis:
        cv2.circle(canvas, (int(kp[0]), int(kp[1])), 3, (255, 255, 255))

      # Draw matches with confidence-based colors
      cmap = matplotlib.cm.get_cmap('turbo')
      for i, (kp0, kp1_vis, conf) in enumerate(zip(kpts0, kpts1_vis, confidence)):
        color = tuple((numpy.array(cmap(conf)) * 255).astype(int).tolist())
        cv2.line(canvas, (int(kp0[0]), int(kp0[1])), (int(kp1_vis[0]), int(kp1_vis[1])), color, 2)

      cv2.imwrite('%s/%s_sift.png' % (data_path, bag_name), canvas)

      if opt.show_keypoints:
        cv2.imshow('SIFT Matches', canvas)
        cv2.waitKey(0)
    else:
      # Create empty visualization
      camera_canvas = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.cvtColor(lidar_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.resize(lidar_canvas, (camera_image.shape[1], camera_image.shape[0]))
      canvas = numpy.concatenate([camera_canvas, lidar_canvas], axis=1)
      cv2.imwrite('%s/%s_sift.png' % (data_path, bag_name), canvas)

if __name__ == '__main__':
  main() 