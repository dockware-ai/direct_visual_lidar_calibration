#!/usr/bin/env python3
# XFeat + LightGlue based correspondence matching alternative to SuperGlue
import sys
import cv2
import math
import json
import torch
import numpy
import argparse
import matplotlib
import psutil
import gc
import os

try:
    # XFeat for feature detection and description  
    from xfeat import XFeat
    # LightGlue for matching
    from lightglue import LightGlue, SuperPoint, DISK, SIFT as LightGlueSIFT
    from lightglue.utils import load_image, rbd
except ImportError as e:
    print(f"Error: XFeat and LightGlue not installed: {e}")
    print("Please install with:")
    print("  git clone https://github.com/cvg/LightGlue.git && cd LightGlue && pip install -e .")
    print("  git clone https://github.com/verlab/accelerated_features.git && cd accelerated_features && pip install -e .")
    sys.exit(1)

def main():
  print('\033[92m' + '*******************************************************************************************' + '\033[0m')
  print('\033[92m' + '* Using XFeat + LightGlue for correspondence matching - Fast and memory efficient       *' + '\033[0m')
  print('\033[92m' + '*******************************************************************************************' + '\033[0m')

  parser = argparse.ArgumentParser(description='Feature matching using XFeat + LightGlue', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('data_path', help='Input data path')
  parser.add_argument('--matcher_type', choices={'xfeat', 'superpoint', 'disk', 'sift'}, default='xfeat', help='Feature detector/descriptor type')
  parser.add_argument('--confidence_threshold', type=float, default=0.2, help='Confidence threshold for filtering matches')
  parser.add_argument('--max_keypoints', type=int, default=4096, help='Maximum number of keypoints to extract')
  parser.add_argument('--min_matches', type=int, default=15, help='Minimum number of matches required')
  parser.add_argument('--show_keypoints', action='store_true', help='Show the detected keypoints')
  parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')
  parser.add_argument('--rotate_camera', type=int, default=0, help='Rotate camera image before matching (0, 90, 180, or 270) (CW)')
  parser.add_argument('--rotate_lidar', type=int, default=0, help='Rotate LiDAR image before matching (0, 90, 180, or 270) (CW)')

  opt = parser.parse_args()
  print(opt)
  
  # Early system info
  print(f"System memory info at startup:")
  mem = psutil.virtual_memory()
  print(f"  Total memory: {mem.total / (1024**3):.2f} GB")
  print(f"  Available memory: {mem.available / (1024**3):.2f} GB")
  print(f"  Memory usage: {mem.percent}%")
  print(f"  CPU count: {psutil.cpu_count()}")
  
  torch.set_grad_enabled(False)
  device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
  print('Running inference on device \"{}\"'.format(device))
  
  # Set PyTorch to use single thread for better stability
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

  def preprocess_image_for_xfeat(image):
    """Convert image to format expected by XFeat/LightGlue"""
    if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to tensor and normalize to [0, 1]
    tensor = torch.from_numpy(image).float() / 255.0
    # XFeat expects [1, 1, H, W] format
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor.to(device)

  def preprocess_image_for_lightglue(image):
    """Convert image to format expected by LightGlue extractors"""
    if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
      # Convert grayscale to RGB for SuperPoint/DISK
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert to tensor [C, H, W] format and normalize
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)  # Add batch dimension [1, C, H, W]

  data_path = opt.data_path
  with open(data_path + '/calib.json', 'r') as f:
    calib_config = json.load(f)

  # Initialize feature extractor and matcher
  print(f"Initializing {opt.matcher_type} feature extractor and LightGlue matcher...")
  print(f"Memory before model loading: {psutil.virtual_memory().percent}%")
  
  try:
    if opt.matcher_type == 'xfeat':
      # Use XFeat for feature detection and description
      print("Loading XFeat feature extractor...")
      extractor = XFeat()
      if hasattr(extractor, 'to'):
        extractor = extractor.to(device)
      
      # LightGlue matcher for XFeat features
      print("Loading LightGlue matcher for XFeat...")
      matcher = LightGlue(features='xfeat').eval().to(device)
      use_xfeat = True
      
    elif opt.matcher_type == 'superpoint':
      print("Loading SuperPoint extractor...")
      extractor = SuperPoint(max_num_keypoints=opt.max_keypoints).eval().to(device)
      print("Loading LightGlue matcher for SuperPoint...")
      matcher = LightGlue(features='superpoint').eval().to(device)
      use_xfeat = False
      
    elif opt.matcher_type == 'disk':
      print("Loading DISK extractor...")
      extractor = DISK(max_num_keypoints=opt.max_keypoints).eval().to(device)
      print("Loading LightGlue matcher for DISK...")
      matcher = LightGlue(features='disk').eval().to(device)
      use_xfeat = False
      
    elif opt.matcher_type == 'sift':
      print("Loading SIFT extractor...")
      extractor = LightGlueSIFT(max_num_keypoints=opt.max_keypoints).eval().to(device)
      print("Loading LightGlue matcher for SIFT...")
      matcher = LightGlue(features='sift').eval().to(device)
      use_xfeat = False
    
    print(f"Successfully loaded {opt.matcher_type} + LightGlue")
    print(f"Memory after model loading: {psutil.virtual_memory().percent}%")
    
  except Exception as e:
    print(f"Error loading {opt.matcher_type} + LightGlue: {e}")
    import traceback
    traceback.print_exc()
    print("Falling back to OpenCV SIFT...")
    extractor = None
    matcher = None
    use_xfeat = False

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
    else:
      print(f"Loaded camera image: {camera_image.shape}")
    
    print(f"Loading lidar image: {lidar_path}")
    lidar_image = cv2.imread(lidar_path, 0)
    if lidar_image is None:
      print(f'ERROR: Could not load lidar image {lidar_path}')
      continue
    else:
      print(f"Loaded lidar image: {lidar_image.shape}")

    # Apply rotations if requested
    if opt.rotate_camera:
      code, camera_R_inv = angle_to_rot(opt.rotate_camera, camera_image.shape)
      camera_image = cv2.rotate(camera_image, code)
    if opt.rotate_lidar:
      code, lidar_R_inv = angle_to_rot(opt.rotate_lidar, lidar_image.shape)
      lidar_image = cv2.rotate(lidar_image, code)

    try:
      if extractor is not None and matcher is not None:
        # Use XFeat/LightGlue pipeline
        print(f"Preprocessing images for {opt.matcher_type}...")
        
        if use_xfeat:
          # XFeat expects grayscale images
          img0_tensor = preprocess_image_for_xfeat(camera_image)
          img1_tensor = preprocess_image_for_xfeat(lidar_image)
          
          print(f"Extracting XFeat features...")
          print(f"Memory before feature extraction: {psutil.virtual_memory().percent}%")
          
          # Extract features with XFeat
          with torch.no_grad():
            # XFeat API might vary, try different methods
            try:
              output0 = extractor.detectAndCompute(img0_tensor, top_k=opt.max_keypoints)
              output1 = extractor.detectAndCompute(img1_tensor, top_k=opt.max_keypoints)
            except Exception as e:
              print(f"XFeat detectAndCompute failed: {e}, trying alternative API...")
              # Alternative API format
              output0 = extractor(img0_tensor)
              output1 = extractor(img1_tensor)
          
          print(f"Memory after feature extraction: {psutil.virtual_memory().percent}%")
          
        else:
          # LightGlue extractors expect RGB images
          img0_tensor = preprocess_image_for_lightglue(camera_image)
          img1_tensor = preprocess_image_for_lightglue(lidar_image)
          
          print(f"Extracting {opt.matcher_type} features...")
          print(f"Memory before feature extraction: {psutil.virtual_memory().percent}%")
          
          # Extract features
          with torch.no_grad():
            output0 = extractor({'image': img0_tensor})
            output1 = extractor({'image': img1_tensor})
          
          print(f"Memory after feature extraction: {psutil.virtual_memory().percent}%")
        
        # Match features with LightGlue
        print(f"Matching features with LightGlue...")
        print(f"Memory before matching: {psutil.virtual_memory().percent}%")
        
        with torch.no_grad():
          matches01 = matcher({'image0': output0, 'image1': output1})
        
        print(f"Memory after matching: {psutil.virtual_memory().percent}%")
        
        # Extract matched keypoints
        if use_xfeat:
          kpts0 = output0['keypoints'][0].cpu().numpy()  # XFeat format
          kpts1 = output1['keypoints'][0].cpu().numpy()
        else:
          kpts0 = output0['keypoints'][0].cpu().numpy()  # LightGlue format
          kpts1 = output1['keypoints'][0].cpu().numpy()
        
        matches = matches01['matches'][0].cpu().numpy()
        confidence = matches01['matching_scores'][0].cpu().numpy()
        
        # Filter valid matches
        valid = matches > -1
        matches = matches[valid]
        confidence = confidence[valid]
        
        # Filter by confidence threshold
        conf_mask = confidence > opt.confidence_threshold
        matches = matches[conf_mask]
        confidence = confidence[conf_mask]
        
        # Extract matched keypoint coordinates
        kpts0_matched = kpts0[conf_mask]
        kpts1_matched = kpts1[matches[conf_mask]]
        
        # Create matches array
        match_indices = numpy.arange(len(kpts0_matched))
        
        print(f'Found {len(kpts0_matched)} matches for {bag_name}')
        
      else:
        # Fallback to OpenCV SIFT
        print("Using OpenCV SIFT fallback")
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
        
        kpts0_matched = numpy.array([kp1[m.queryIdx].pt for m in good_matches], dtype=numpy.float32)
        kpts1_matched = numpy.array([kp2[m.trainIdx].pt for m in good_matches], dtype=numpy.float32)
        
        match_indices = numpy.arange(len(good_matches))
        distances = numpy.array([m.distance for m in good_matches])
        confidence = 1.0 / (1.0 + distances)
        if len(confidence) > 0:
          confidence = confidence / numpy.max(confidence)

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
      'confidence': confidence.flatten().tolist() if len(confidence) > 0 else []
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
        cv2.circle(canvas, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
      for kp in kpts1_vis:
        cv2.circle(canvas, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

      # Draw matches with confidence-based colors
      cmap = matplotlib.cm.get_cmap('plasma')
      normalized_confidence = confidence / numpy.max(confidence) if len(confidence) > 0 else confidence
      
      for i, (kp0, kp1_vis, conf) in enumerate(zip(kpts0_matched, kpts1_vis, normalized_confidence)):
        color = tuple((numpy.array(cmap(conf)) * 255).astype(int).tolist()[:3])
        cv2.line(canvas, (int(kp0[0]), int(kp0[1])), (int(kp1_vis[0]), int(kp1_vis[1])), color, 2)

      cv2.imwrite('%s/%s_xfeat.png' % (data_path, bag_name), canvas)

      if opt.show_keypoints:
        cv2.imshow(f'{opt.matcher_type} + LightGlue Matches', canvas)
        cv2.waitKey(0)
    else:
      # Create empty visualization
      camera_canvas = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.cvtColor(lidar_image, cv2.COLOR_GRAY2BGR)
      lidar_canvas = cv2.resize(lidar_canvas, (camera_image.shape[1], camera_image.shape[0]))
      canvas = numpy.concatenate([camera_canvas, lidar_canvas], axis=1)
      cv2.imwrite('%s/%s_xfeat.png' % (data_path, bag_name), canvas)

if __name__ == '__main__':
  main() 