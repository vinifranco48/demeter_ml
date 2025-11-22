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

def extract_shape_features(contour: np.ndarray) -> np.ndarray:
    """
    Extracts shape features: Area, Perimeter, Circularity, Aspect Ratio.
    Uses Rotated Rectangle (minAreaRect) for accurate dimensions.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        circularity = 0
    else:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
    # Use minAreaRect to get accurate dimensions regardless of rotation
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect
    
    # Ensure w is the smaller dimension (width) and h is larger (length)
    if w > h:
        w, h = h, w
        
    aspect_ratio = float(w) / h if h != 0 else 0
    
    return np.array([area, perimeter, circularity, aspect_ratio])

def extract_all_features(grain_image: np.ndarray) -> np.ndarray:
    """
    Wrapper to extract all features from a cropped grain image.
    Re-calculates contour from the crop for shape features.
    """
    gray = cv2.cvtColor(grain_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros(13) # 9 color + 4 shape
        
    # Take the largest contour in the crop (should be the grain)
    contour = max(contours, key=cv2.contourArea)
    
    color_feats = extract_color_features(grain_image)
    shape_feats = extract_shape_features(contour)
    
    return np.concatenate([color_feats, shape_feats])
