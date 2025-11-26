import cv2
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from demeter_ml.processing import segment_grains, sharpen_image
from demeter_ml.features import extract_all_features
from demeter_ml.analysis import analyze_grain_rules

def create_synthetic_image():
    # Create a black image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 1. Two touching circles (White) - Should be separated by Watershed
    cv2.circle(img, (100, 100), 40, (200, 200, 200), -1)
    cv2.circle(img, (160, 100), 40, (200, 200, 200), -1)
    
    # 2. A circle with a hole (White with Black hole) - Should be detected as Perforated
    cv2.circle(img, (100, 300), 40, (200, 200, 200), -1)
    cv2.circle(img, (100, 300), 10, (0, 0, 0), -1)
    
    # 3. A non-convex shape (Pacman-like) - Should be detected as Broken/Chipped
    cv2.ellipse(img, (300, 100), (40, 40), 0, 45, 315, (200, 200, 200), -1)
    
    # 4. A rough texture grain (Noise) - Should have high entropy
    cv2.circle(img, (300, 300), 40, (200, 200, 200), -1)
    noise = np.random.randint(0, 50, (400, 400, 3), dtype=np.uint8)
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(mask, (300, 300), 40, 255, -1)
    img = cv2.add(img, cv2.bitwise_and(noise, noise, mask=mask))
    
    return img

def test_pipeline():
    print("Creating synthetic image...")
    img = create_synthetic_image()
    cv2.imwrite("synthetic_test.jpg", img)
    
    print("Running pipeline...")
    sharpened = sharpen_image(img)
    grain_masks, markers, contours, grain_holes = segment_grains(sharpened)
    
    print(f"Found {len(grain_masks)} grains.")
    
    # We expect 4 grains:
    # 1. Left of touching pair
    # 2. Right of touching pair
    # 3. Perforated grain
    # 4. Non-convex grain
    # 5. Rough grain? Wait, I drew 4 items.
    # Touching pair is 2 items.
    # Perforated is 1 item.
    # Non-convex is 1 item.
    # Rough is 1 item.
    # Total 5 grains expected if touching are separated.
    
    for i, (grain, contour, holes) in enumerate(zip(grain_masks, contours, grain_holes)):
        feats = extract_all_features(grain, contour=None)
        analysis = analyze_grain_rules(feats, holes=holes)
        
        print(f"Grain {i}:")
        print(f"  Classification: {analysis['classification']}")
        print(f"  Reasons: {analysis['reasons']}")
        print(f"  Convexity: {analysis['features']['convexity']:.2f}")
        print(f"  Entropy: {analysis['features']['entropy']:.2f}")
        print(f"  Holes: {len(holes)}")
        print("-" * 20)

if __name__ == "__main__":
    test_pipeline()
