import numpy as np

def analyze_grain_rules(features: np.ndarray) -> dict:
    """
    Analyzes the suitability of a grain based on fixed rules (heuristics).
    """
    # Unpack features
    # 0-2: BGR, 3-5: HSV, 6-8: LAB
    # 9: Area, 10: Perimeter, 11: Circularity, 12: AspectRatio
    
    mean_h = features[3]
    mean_s = features[4]
    mean_v = features[5]
    circularity = features[11]
    aspect_ratio = features[12]
    
    classification = "Good"
    reasons = []
    
    # 1. Shape Analysis (Soybeans should be round)
    if circularity < 0.70:
        classification = "Defect: Broken/Irregular"
        reasons.append(f"Low circularity ({circularity:.2f})")
        
    if aspect_ratio < 0.75 or aspect_ratio > 1.25:
        if "Defect: Broken/Irregular" not in classification:
             classification = "Defect: Elongated"
        reasons.append(f"Bad aspect ratio ({aspect_ratio:.2f})")

    # 2. Color Analysis
    # Dark spots / Rotten (Low Value)
    if mean_v < 60:
        classification = "Defect: Dark/Rotten"
        reasons.append(f"Too dark (V={mean_v:.1f})")
        
    # Green / Immature (Hue approx 30-90 for Green in OpenCV)
    # Yellow/Orange is typically 15-30.
    if 35 < mean_h < 85:
        classification = "Defect: Green/Immature"
        reasons.append(f"Greenish color (H={mean_h:.1f})")
        
    return {
        "classification": classification,
        "reasons": reasons,
        "features": {
            "mean_h": float(mean_h),
            "mean_v": float(mean_v),
            "circularity": float(circularity)
        }
    }
