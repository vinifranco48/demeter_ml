import cv2
import numpy as np

def extract_color_features(grain_image: np.ndarray) -> np.ndarray:
    """
    Extracts mean color features (B, G, R, H, S, V, L, A, B) from a segmented grain image.
    Assumes the background is black (0).
    """
    # Create a mask for non-black pixels
    gray = cv2.cvtColor(grain_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    if cv2.countNonZero(mask) == 0:
        return np.zeros(9)

    # BGR Mean
    mean_bgr = cv2.mean(grain_image, mask=mask)[:3]
    
    # HSV Mean
    hsv = cv2.cvtColor(grain_image, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
    
    # LAB Mean
    lab = cv2.cvtColor(grain_image, cv2.COLOR_BGR2LAB)
    mean_lab = cv2.mean(lab, mask=mask)[:3]
    
    return np.concatenate([mean_bgr, mean_hsv, mean_lab])

def extract_texture_features(grain_image: np.ndarray) -> np.ndarray:
    """
    Extracts texture features using a simplified Local Binary Pattern (LBP) approach.
    Returns: Mean LBP, Std LBP, Entropy.
    """
    gray = cv2.cvtColor(grain_image, cv2.COLOR_BGR2GRAY)
    
    # Simple LBP implementation
    # Compare each pixel with its 8 neighbors
    rows, cols = gray.shape
    lbp_image = np.zeros_like(gray)
    
    # Vectorized approach for 3x3 LBP
    # Pad image to handle borders
    padded = np.pad(gray, ((1,1), (1,1)), mode='reflect')
    
    center = padded[1:-1, 1:-1]
    
    # Neighbors positions relative to center
    shifts = [(-1, -1), (-1, 0), (-1, 1),
              (0, 1), (1, 1), (1, 0),
              (1, -1), (0, -1)]
              
    power_val = 1
    for dy, dx in shifts:
        neighbor = padded[1+dy:1+dy+rows, 1+dx:1+dx+cols]
        lbp_image += ((neighbor >= center) * power_val).astype(np.uint8)
        power_val *= 2
        
    # Calculate statistics on LBP image (ignoring background 0 if masked, but here we have crop)
    # We should mask out the background.
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    if cv2.countNonZero(mask) == 0:
        return np.zeros(3)
        
    lbp_values = lbp_image[mask > 0]
    
    mean_lbp = np.mean(lbp_values)
    std_lbp = np.std(lbp_values)
    
    # Entropy
    hist, _ = np.histogram(lbp_values, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    return np.array([mean_lbp, std_lbp, entropy])

def extract_shape_features(contour: np.ndarray) -> np.ndarray:
    """
    Extracts shape features: Area, Perimeter, Circularity, Aspect Ratio, Convexity, Solidity.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        circularity = 0
    else:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
    # Aspect Ratio
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect
    if w > h: w, h = h, w
    aspect_ratio = float(w) / h if h != 0 else 0
    
    # Convexity
    is_convex = cv2.isContourConvex(contour)
    convexity = 1.0 if is_convex else 0.0
    
    # Solidity (Area / Convex Hull Area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    return np.array([area, perimeter, circularity, aspect_ratio, convexity, solidity])

def extract_all_features(grain_image: np.ndarray, contour: np.ndarray = None) -> np.ndarray:
    """
    Wrapper to extract all features from a cropped grain image.
    If contour is provided, uses it directly. Otherwise, finds it in the crop.
    """
    if contour is None:
        gray = cv2.cvtColor(grain_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(18) # 9 color + 6 shape + 3 texture
            
        contour = max(contours, key=cv2.contourArea)
    
    color_feats = extract_color_features(grain_image)
    shape_feats = extract_shape_features(contour)
    texture_feats = extract_texture_features(grain_image)
    
    return np.concatenate([color_feats, shape_feats, texture_feats])
