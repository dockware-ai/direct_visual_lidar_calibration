#!/usr/bin/env python3
# Simple PyTorch-based correspondence matching using basic features
import sys
import cv2
import json
import torch
import numpy
import argparse
import matplotlib
import psutil
import torch.nn.functional as F

def main():
  print('\033[93m' + '*******************************************************************************************' + '\033[0m')
  print('\033[93m' + '* Simple PyTorch feature matching - Lightweight and memory efficient                   *' + '\033[0m')
  print('\033[93m' + '*******************************************************************************************' + '\033[0m')

  parser = argparse.ArgumentParser(description='Simple feature matching using PyTorch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('data_path', help='Input data path')
  parser.add_argument('--confidence_threshold', type=float, default=0.1, help='Confidence threshold for filtering matches')
  parser.add_argument('--max_keypoints', type=int, default=2048, help='Maximum number of keypoints to extract')
  parser.add_argument('--min_matches', type=int, default=15, help='Minimum number of matches required')
  parser.add_argument('--show_keypoints', action='store_true', help='Show the detected keypoints')
  parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')
  parser.add_argument('--rotate_camera', type=int, default=0, help='Rotate camera image before matching (0, 90, 180, or 270) (CW)')
  parser.add_argument('--rotate_lidar', type=int, default=0, help='Rotate LiDAR image before matching (0, 90, 180, or 270) (CW)')

  opt = parser.parse_args()
  print(opt)
  
  # System info
  print(f"System memory info at startup:")
  mem = psutil.virtual_memory()
  print(f"  Total memory: {mem.total / (1024**3):.2f} GB")
  print(f"  Available memory: {mem.available / (1024**3):.2f} GB")
  print(f"  Memory usage: {mem.percent}%")
  
  torch.set_grad_enabled(False)
  device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
  print('Running inference on device \"{}\"'.format(device))
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

  def simple_feature_extractor(image_tensor, max_keypoints=2048):
    """Simple PyTorch-based feature extractor using Harris corners and descriptors"""
    # Convert to grayscale if needed
    if image_tensor.dim() == 4 and image_tensor.size(1) == 3:
      image_tensor = torch.mean(image_tensor, dim=1, keepdim=True)
    
    # Harris corner detection using PyTorch
    # Sobel filters for gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    
    # Calculate gradients
    Ix = F.conv2d(image_tensor, sobel_x, padding=1)
    Iy = F.conv2d(image_tensor, sobel_y, padding=1)
    
    # Harris matrix components
    Ixx = Ix * Ix
    Iyy = Iy * Iy  
    Ixy = Ix * Iy
    
    # Gaussian smoothing
    gaussian = torch.ones(1, 1, 5, 5, device=device) / 25.0
    Sxx = F.conv2d(Ixx, gaussian, padding=2)
    Syy = F.conv2d(Iyy, gaussian, padding=2)
    Sxy = F.conv2d(Ixy, gaussian, padding=2)
    
    # Harris response
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    harris = det - 0.04 * trace * trace
    
    # Non-maximum suppression
    maxpool = F.max_pool2d(harris, kernel_size=7, stride=1, padding=3)
    peaks = (harris == maxpool) & (harris > 0.01)
    
    # Get keypoint coordinates
    keypoints = torch.nonzero(peaks.squeeze(), as_tuple=False).float()
    if len(keypoints) > max_keypoints:
      # Keep strongest keypoints
      values = harris.squeeze()[keypoints[:, 0].long(), keypoints[:, 1].long()]
      _, indices = torch.topk(values, max_keypoints)
      keypoints = keypoints[indices]
    
    # Swap x,y coordinates (row,col -> x,y)
    keypoints = keypoints[:, [1, 0]]
    
    # Extract simple descriptors around keypoints
    descriptors = []
    patch_size = 16
    for kp in keypoints:
      x, y = int(kp[0]), int(kp[1])
      # Bounds checking
      if (x - patch_size//2 >= 0 and x + patch_size//2 < image_tensor.size(3) and
          y - patch_size//2 >= 0 and y + patch_size//2 < image_tensor.size(2)):
        patch = image_tensor[0, 0, y-patch_size//2:y+patch_size//2, x-patch_size//2:x+patch_size//2]
        # Simple descriptor: normalized patch
        desc = patch.flatten()
        desc = desc / (torch.norm(desc) + 1e-8)
        descriptors.append(desc)
      else:
        # Remove keypoint if patch is out of bounds
        continue
    
    if len(descriptors) == 0:
      return torch.empty(0, 2), torch.empty(0, patch_size*patch_size)
    
    descriptors = torch.stack(descriptors)
    valid_keypoints = keypoints[:len(descriptors)]
    
    return valid_keypoints, descriptors

  def match_descriptors(desc1, desc2, threshold=0.8):
    """Simple descriptor matching using nearest neighbor"""
    if len(desc1) == 0 or len(desc2) == 0:
      return torch.empty(0, dtype=torch.long), torch.empty(0)
    
    # Compute distances
    distances = torch.cdist(desc1, desc2)
    
    # Find nearest neighbors
    values, indices = torch.min(distances, dim=1)
    
    # Ratio test (if we have enough descriptors)
    if desc2.size(0) > 1:
      sorted_distances = torch.sort(distances, dim=1)[0]
      ratios = sorted_distances[:, 0] / (sorted_distances[:, 1] + 1e-8)
      valid_mask = ratios < threshold
    else:
      valid_mask = values < 0.5
    
    # Filter matches
    matches = indices[valid_mask]
    confidences = 1.0 - values[valid_mask]
    
    return matches, confidences

  data_path = opt.data_path
  with open(data_path + '/calib.json', 'r') as f:
    calib_config = json.load(f)

  print(f"Simple PyTorch feature extractor loaded")
  print(f"Memory after initialization: {psutil.virtual_memory().percent}%")

  # Process each image pair
  for bag_name in calib_config['meta']['bag_names']:
    print('processing %s' % bag_name)
    
    print(f"Memory before processing {bag_name}: {psutil.virtual_memory().percent}%")

    camera_path = '%s/%s.png' % (data_path, bag_name)
    lidar_path = '%s/%s_lidar_intensities.png' % (data_path, bag_name)
    
    print(f"Loading camera image: {camera_path}")
    camera_image = cv2.imread(camera_path, 0)
    if camera_image is None:
      print(f'ERROR: Could not load camera image {camera_path}')
      continue
    print(f"Loaded camera image: {camera_image.shape}")
    
    print(f"Loading lidar image: {lidar_path}")
    lidar_image = cv2.imread(lidar_path, 0)
    if lidar_image is None:
      print(f'ERROR: Could not load lidar image {lidar_path}')
      continue
    print(f"Loaded lidar image: {lidar_image.shape}")

    # Apply rotations if requested
    if opt.rotate_camera:
      code, camera_R_inv = angle_to_rot(opt.rotate_camera, camera_image.shape)
      camera_image = cv2.rotate(camera_image, code)
    if opt.rotate_lidar:
      code, lidar_R_inv = angle_to_rot(opt.rotate_lidar, lidar_image.shape)
      lidar_image = cv2.rotate(lidar_image, code)

    try:
      # Convert to tensors
      print(f"Converting images to tensors...")
      img0_tensor = torch.from_numpy(camera_image).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
      img1_tensor = torch.from_numpy(lidar_image).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
      
      print(f"Memory after tensor conversion: {psutil.virtual_memory().percent}%")
      
      # Extract features
      print(f"Extracting features from camera image...")
      kpts0, desc0 = simple_feature_extractor(img0_tensor, opt.max_keypoints)
      print(f"Found {len(kpts0)} keypoints in camera image")
      
      print(f"Extracting features from lidar image...")
      kpts1, desc1 = simple_feature_extractor(img1_tensor, opt.max_keypoints)
      print(f"Found {len(kpts1)} keypoints in lidar image")
      
      print(f"Memory after feature extraction: {psutil.virtual_memory().percent}%")
      
      # Match features
      print(f"Matching features...")
      matches, confidences = match_descriptors(desc0, desc1)
      
      print(f"Memory after matching: {psutil.virtual_memory().percent}%")
      
      # Filter by confidence
      conf_mask = confidences > opt.confidence_threshold
      matches = matches[conf_mask]
      confidences = confidences[conf_mask]
      
      # Extract matched keypoints
      if len(matches) > 0:
        kpts0_matched = kpts0[:len(matches)].cpu().numpy()
        kpts1_matched = kpts1[matches].cpu().numpy()
        confidences = confidences.cpu().numpy()
        match_indices = numpy.arange(len(matches))
      else:
        kpts0_matched = numpy.empty((0, 2))
        kpts1_matched = numpy.empty((0, 2))
        confidences = numpy.empty(0)
        match_indices = numpy.empty(0, dtype=int)
      
      print(f'Found {len(kpts0_matched)} matches for {bag_name}')

    except Exception as e:
      print(f'Error during matching for {bag_name}: {e}')
      import traceback
      traceback.print_exc()
      # Create empty results
      result = {'kpts0': [], 'kpts1': [], 'matches': [], 'confidence': []}
      with open('%s/%s_matches.json' % (data_path, bag_name), 'w') as f:
        json.dump(result, f)
      continue

    if len(kpts0_matched) < opt.min_matches:
      print(f'Warning: Not enough matches found in {bag_name} ({len(kpts0_matched)} < {opt.min_matches})')

    # Apply reverse rotations if needed
    kpts0_final = kpts0_matched
    kpts1_final = kpts1_matched

    if opt.rotate_camera and len(kpts0_final) > 0:
      kpts0_final = camera_R_inv(kpts0_final)
    if opt.rotate_lidar and len(kpts1_final) > 0:
      kpts1_final = lidar_R_inv(kpts1_final)

    # Save results
    result = { 
      'kpts0': kpts0_final.flatten().tolist() if len(kpts0_final) > 0 else [], 
      'kpts1': kpts1_final.flatten().tolist() if len(kpts1_final) > 0 else [], 
      'matches': match_indices.flatten().tolist() if len(match_indices) > 0 else [], 
      'confidence': confidences.flatten().tolist() if len(confidences) > 0 else []
    }
    with open('%s/%s_matches.json' % (data_path, bag_name), 'w') as f:
      json.dump(result, f)

    # Visualization
    if len(kpts0_matched) > 0:
      camera_canvas = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.cvtColor(lidar_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.resize(lidar_canvas, (camera_image.shape[1], camera_image.shape[0]))

      sx = camera_image.shape[1] / lidar_image.shape[1]
      sy = camera_image.shape[0] / lidar_image.shape[0]

      # Adjust lidar keypoints for visualization
      kpts1_vis = kpts1_matched.copy()
      kpts1_vis[:, 0] = kpts1_vis[:, 0] * sx + camera_image.shape[1]
      kpts1_vis[:, 1] = kpts1_vis[:, 1] * sy

      canvas = numpy.concatenate([camera_canvas, lidar_canvas], axis=1)
      
      # Draw keypoints
      for kp in kpts0_matched:
        cv2.circle(canvas, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)
      for kp in kpts1_vis:
        cv2.circle(canvas, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)

      # Draw matches with confidence-based colors
      cmap = matplotlib.cm.get_cmap('viridis')
      normalized_confidence = confidences / numpy.max(confidences) if len(confidences) > 0 else confidences
      
      for i, (kp0, kp1_vis, conf) in enumerate(zip(kpts0_matched, kpts1_vis, normalized_confidence)):
        color = tuple((numpy.array(cmap(conf)) * 255).astype(int).tolist()[:3])
        cv2.line(canvas, (int(kp0[0]), int(kp0[1])), (int(kp1_vis[0]), int(kp1_vis[1])), color, 2)

      cv2.imwrite('%s/%s_simple.png' % (data_path, bag_name), canvas)

      if opt.show_keypoints:
        cv2.imshow('Simple PyTorch Matches', canvas)
        cv2.waitKey(0)
    else:
      # Create empty visualization
      camera_canvas = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.cvtColor(lidar_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.resize(lidar_canvas, (camera_image.shape[1], camera_image.shape[0]))
      canvas = numpy.concatenate([camera_canvas, lidar_canvas], axis=1)
      cv2.imwrite('%s/%s_simple.png' % (data_path, bag_name), canvas)

if __name__ == '__main__':
  main() 