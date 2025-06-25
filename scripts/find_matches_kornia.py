#!/usr/bin/env python3
# Kornia LoFTR-based correspondence matching alternative to SuperGlue
import sys
import cv2
import math
import json
import torch
import numpy
import argparse
import matplotlib

try:
    import kornia
    from kornia.feature import LoFTR
    from kornia_moons.feature import *
except ImportError:
    print("Error: Kornia not installed. Please install with: pip install kornia kornia-moons")
    sys.exit(1)

def main():
  print('\033[94m' + '*******************************************************************************************' + '\033[0m')
  print('\033[94m' + '* Using Kornia LoFTR for correspondence matching - Free for academic and commercial use *' + '\033[0m')
  print('\033[94m' + '*******************************************************************************************' + '\033[0m')

  parser = argparse.ArgumentParser(description='Initial guess estimation based on Kornia LoFTR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('data_path', help='Input data path')
  parser.add_argument('--model_type', choices={'indoor', 'outdoor'}, default='outdoor', help='LoFTR model type')
  parser.add_argument('--confidence_threshold', type=float, default=0.2, help='Confidence threshold for filtering matches')
  parser.add_argument('--max_keypoints', type=int, default=2048, help='Maximum number of keypoints to extract')
  parser.add_argument('--min_matches', type=int, default=20, help='Minimum number of matches required')
  parser.add_argument('--show_keypoints', action='store_true', help='Show the detected keypoints')
  parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
  parser.add_argument('--rotate_camera', type=int, default=0, help='Rotate camera image before matching (CW 0, 90, 180, or 270) (CW)')
  parser.add_argument('--rotate_lidar', type=int, default=0, help='Rotate LiDAR image before matching (0, 90, 180, or 270) (CW)')

  opt = parser.parse_args()
  print(opt)

  torch.set_grad_enabled(False)
  
  # Disable NNPACK to avoid ARM64 compatibility issues
  import os
  os.environ['PYTORCH_NNPACK_DISABLE'] = '1'
  
  device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
  print('Running inference on device \"{}\"'.format(device))
  
  # Set PyTorch to use single thread for better stability in Docker
  torch.set_num_threads(1)

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

  def preprocess_image(image):
    """Convert image to tensor and normalize"""
    if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to tensor and normalize to [0, 1]
    tensor = torch.from_numpy(image).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return tensor.to(device)

  data_path = opt.data_path
  with open(data_path + '/calib.json', 'r') as f:
    calib_config = json.load(f)

  # Initialize LoFTR matcher with better error handling
  use_sift_fallback = False
  try:
    print(f"Loading LoFTR model ({opt.model_type})...")
    if opt.model_type == 'outdoor':
      matcher = LoFTR(pretrained='outdoor').eval()
    else:
      matcher = LoFTR(pretrained='indoor').eval()
    
    # Move to device after loading to avoid memory issues
    matcher = matcher.to(device)
    print(f"Successfully loaded LoFTR model ({opt.model_type})")
    
  except Exception as e:
    print(f"Error loading LoFTR model: {e}")
    print("Falling back to basic Kornia SIFT feature matching...")
    
    # Fallback to basic Kornia SIFT-like features if LoFTR fails
    try:
      from kornia.feature import SIFTFeature
      matcher = SIFTFeature(num_features=opt.max_keypoints).to(device)
      use_sift_fallback = True
      print("Successfully loaded SIFT fallback")
    except Exception as e2:
      print(f"Error loading SIFT fallback: {e2}")
      print("Using OpenCV SIFT as final fallback")
      matcher = None
      use_sift_fallback = True

  for bag_name in calib_config['meta']['bag_names']:
    print('processing %s' % bag_name)

    camera_image = cv2.imread('%s/%s.png' % (data_path, bag_name), 0)
    lidar_image = cv2.imread('%s/%s_lidar_intensities.png' % (data_path, bag_name), 0)

    if camera_image is None or lidar_image is None:
      print(f'Warning: Could not load images for {bag_name}')
      continue

    if opt.rotate_camera:
      code, camera_R_inv = angle_to_rot(opt.rotate_camera, camera_image.shape)
      camera_image = cv2.rotate(camera_image, code)
    if opt.rotate_lidar:
      code, lidar_R_inv = angle_to_rot(opt.rotate_lidar, lidar_image.shape)
      lidar_image = cv2.rotate(lidar_image, code)

    # Preprocess images
    img0_tensor = preprocess_image(camera_image)
    img1_tensor = preprocess_image(lidar_image)

    try:
      if use_sift_fallback and matcher is not None:
        # Use Kornia SIFT-like features
        lafs0, resps0, descs0 = matcher(img0_tensor)
        lafs1, resps1, descs1 = matcher(img1_tensor)
        
        # Simple nearest neighbor matching
        from kornia.feature import match_nn
        dists, indices = match_nn(descs0, descs1)
        
        # Filter matches by distance threshold
        mask = dists < opt.confidence_threshold
        indices = indices[mask]
        
        # Extract keypoints
        kpts0 = kornia.feature.laf.get_laf_center(lafs0)[0][mask].cpu().numpy()
        kpts1 = kornia.feature.laf.get_laf_center(lafs1)[0][indices].cpu().numpy()
        
        # Create confidence scores
        confidence = (1.0 - dists[mask]).cpu().numpy()
        matches = numpy.arange(len(kpts0))
        
      elif use_sift_fallback and matcher is None:
        # Final fallback: use OpenCV SIFT
        print("Using OpenCV SIFT as final fallback")
        sift = cv2.SIFT_create(nfeatures=opt.max_keypoints)
        
        kp1, des1 = sift.detectAndCompute(camera_image, None)
        kp2, des2 = sift.detectAndCompute(lidar_image, None)
        
        if des1 is None or des2 is None:
          raise Exception("No features detected with OpenCV SIFT")
        
        # FLANN matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches_raw = flann.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches_raw:
          if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
              good_matches.append(m)
        
        kpts0 = numpy.array([kp1[m.queryIdx].pt for m in good_matches], dtype=numpy.float32)
        kpts1 = numpy.array([kp2[m.trainIdx].pt for m in good_matches], dtype=numpy.float32)
        
        matches = numpy.arange(len(good_matches))
        distances = numpy.array([m.distance for m in good_matches])
        confidence = 1.0 / (1.0 + distances)
        if len(confidence) > 0:
          confidence = confidence / numpy.max(confidence)
        
      else:
        # Use LoFTR
        input_dict = {
          'image0': img0_tensor,
          'image1': img1_tensor
        }
        
        correspondences = matcher(input_dict)
        
        kpts0 = correspondences['keypoints0'].cpu().numpy()
        kpts1 = correspondences['keypoints1'].cpu().numpy()
        confidence = correspondences['confidence'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = confidence > opt.confidence_threshold
        kpts0 = kpts0[mask]
        kpts1 = kpts1[mask]
        confidence = confidence[mask]
        
        # Create matches array
        matches = numpy.arange(len(kpts0))

    except Exception as e:
      print(f'Error during matching for {bag_name}: {e}')
      # Create empty results
      result = {'kpts0': [], 'kpts1': [], 'matches': [], 'confidence': []}
      with open('%s/%s_matches.json' % (data_path, bag_name), 'w') as f:
        json.dump(result, f)
      continue

    print(f'Found {len(kpts0)} matches for {bag_name}')

    if len(kpts0) < opt.min_matches:
      print(f'Warning: Not enough matches found in {bag_name} ({len(kpts0)} < {opt.min_matches})')

    kpts0_ = kpts0
    kpts1_ = kpts1

    if opt.rotate_camera and len(kpts0_) > 0:
      kpts0_ = camera_R_inv(kpts0_)
    if opt.rotate_lidar and len(kpts1_) > 0:
      kpts1_ = lidar_R_inv(kpts1_)

    result = { 
      'kpts0': kpts0_.flatten().tolist() if len(kpts0_) > 0 else [], 
      'kpts1': kpts1_.flatten().tolist() if len(kpts1_) > 0 else [], 
      'matches': matches.flatten().tolist() if len(matches) > 0 else [], 
      'confidence': confidence.flatten().tolist() if len(confidence) > 0 else []
    }
    with open('%s/%s_matches.json' % (data_path, bag_name), 'w') as f:
      json.dump(result, f)

    # Visualization
    if len(kpts0) > 0:
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
      normalized_confidence = confidence / numpy.max(confidence) if len(confidence) > 0 else confidence
      
      for i, (kp0, kp1_vis, conf) in enumerate(zip(kpts0, kpts1_vis, normalized_confidence)):
        color = tuple((numpy.array(cmap(conf)) * 255).astype(int).tolist())
        cv2.line(canvas, (int(kp0[0]), int(kp0[1])), (int(kp1_vis[0]), int(kp1_vis[1])), color, 2)

      cv2.imwrite('%s/%s_kornia.png' % (data_path, bag_name), canvas)

      if opt.show_keypoints:
        cv2.imshow('Kornia LoFTR Matches', canvas)
        cv2.waitKey(0)
    else:
      # Create empty visualization
      camera_canvas = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.cvtColor(lidar_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.resize(lidar_canvas, (camera_image.shape[1], camera_image.shape[0]))
      canvas = numpy.concatenate([camera_canvas, lidar_canvas], axis=1)
      cv2.imwrite('%s/%s_kornia.png' % (data_path, bag_name), canvas)

if __name__ == '__main__':
  main() 