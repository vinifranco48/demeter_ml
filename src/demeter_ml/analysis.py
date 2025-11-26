import numpy as np

def is_uncertain_case(features: np.ndarray, classification: str, reasons: list) -> bool:
    """
    Determines if a grain analysis is uncertain and would benefit from LLM verification.
    Returns True for borderline cases where rules are near thresholds.
    """
    # Extract key features
    circularity = features[11]
    aspect_ratio = features[12]
    convexity = features[13]
    mean_v = features[5]
    mean_h = features[3]
    entropy = features[17]

    # Define uncertainty margins (±10% of threshold)
    uncertain = False

    # Borderline circularity (near 0.70 threshold)
    if 0.65 <= circularity <= 0.75:
        uncertain = True

    # Borderline aspect ratio (near 0.75 or 1.25 thresholds)
    if (0.70 <= aspect_ratio <= 0.80) or (1.20 <= aspect_ratio <= 1.30):
        uncertain = True

    # Borderline brightness (near 60 threshold)
    if 55 <= mean_v <= 65:
        uncertain = True

    # Borderline green detection (near hue thresholds 35-85)
    if (30 <= mean_h <= 40) or (80 <= mean_h <= 90):
        uncertain = True

    # Multiple but weak defects (suggests edge case)
    if len(reasons) >= 2 and classification != "Good":
        uncertain = True

    # Good with no reasons but close to thresholds
    if classification == "Good" and len(reasons) == 0:
        if (circularity < 0.80 or aspect_ratio < 0.85 or
            aspect_ratio > 1.15 or mean_v < 80):
            uncertain = True

    return uncertain


def analyze_grain_rules(features: np.ndarray, holes: list = None) -> dict:
    """
    Analyzes the suitability of a grain based on fixed rules (heuristics).
    Features indices:
    0-2: BGR, 3-5: HSV, 6-8: LAB
    9: Area, 10: Perimeter, 11: Circularity, 12: AspectRatio, 13: Convexity, 14: Solidity
    15: Mean LBP, 16: Std LBP, 17: Entropy
    """
    
    mean_h = features[3]
    mean_s = features[4]
    mean_v = features[5]
    circularity = features[11]
    aspect_ratio = features[12]
    convexity = features[13]
    solidity = features[14]
    entropy = features[17]
    
    classification = "Good"
    reasons = []
    
    # 0. Structural Damage (Holes)
    if holes and len(holes) > 0:
        classification = "Defect: Perforated/Damaged"
        reasons.append(f"Has {len(holes)} holes")

    # 1. Shape Analysis
    if circularity < 0.70:
        if "Defect" not in classification: classification = "Defect: Broken/Irregular"
        reasons.append(f"Low circularity ({circularity:.2f})")
        
    if aspect_ratio < 0.75 or aspect_ratio > 1.25:
        if "Defect" not in classification: classification = "Defect: Elongated"
        reasons.append(f"Bad aspect ratio ({aspect_ratio:.2f})")
        
    if convexity < 0.95:
        if "Defect" not in classification: classification = "Defect: Broken/Chipped"
        reasons.append(f"Not convex ({convexity:.2f})")

    # 2. Color Analysis
    # Dark spots / Rotten (Low Value)
    if mean_v < 60:
        if "Defect" not in classification: classification = "Defect: Dark/Rotten"
        reasons.append(f"Too dark (V={mean_v:.1f})")
        
    # Green / Immature (Hue approx 30-90 for Green in OpenCV)
    if 35 < mean_h < 85:
        if "Defect" not in classification: classification = "Defect: Green/Immature"
        reasons.append(f"Greenish color (H={mean_h:.1f})")
        
    # 3. Texture Analysis
    # High entropy might indicate rough surface (wrinkled/diseased)
    if entropy > 5.0: # Threshold needs calibration
        if "Defect" not in classification: classification = "Defect: Rough Texture"
        reasons.append(f"High entropy ({entropy:.2f})")
        
    # Check if this is an uncertain case
    uncertain = is_uncertain_case(features, classification, reasons)

    return {
        "classification": classification,
        "reasons": reasons,
        "uncertain": uncertain,
        "features": {
            "mean_h": float(mean_h),
            "mean_v": float(mean_v),
            "circularity": float(circularity),
            "convexity": float(convexity),
            "entropy": float(entropy)
        }
    }
