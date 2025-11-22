import argparse
import cv2
import numpy as np
import os
from .processing import load_image, preprocess_image, segment_grains
from .features import extract_all_features
from .analysis import analyze_grain_rules

def main():
    parser = argparse.ArgumentParser(description="Grain Classification System")
    parser.add_argument("image_path", help="Path to the image to analyze")
    parser.add_argument("--train", action="store_true", help="Train the model on the input image (unsupervised)")
    parser.add_argument("--model", default="kmeans_model.pkl", help="Path to save/load the model")
    
    args = parser.parse_args()
    
    try:
        image = load_image(args.image_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Preprocessing image...")
    processed_image = preprocess_image(image)
    
    print("Segmenting grains...")
    grain_masks, _ = segment_grains(processed_image)
    print(f"Found {len(grain_masks)} grains.")
    
    if len(grain_masks) == 0:
        print("No grains found.")
        return

    print("Extracting features and analyzing...")
    
    results = []
    all_features = []
    for i, grain in enumerate(grain_masks):
        feats = extract_all_features(grain)
        all_features.append(feats)
        analysis = analyze_grain_rules(feats)
        results.append(analysis)
        print(f"Grain {i}: {analysis['classification']} - {analysis['reasons']}")

    # Summary
    good_count = sum(1 for r in results if r['classification'] == "Good")
    print(f"\nSummary: {good_count}/{len(grain_masks)} grains are Good.")

    print("Generating result image...")
    result_image = image.copy()
    
    # Save features to CSV
    import csv
    with open("features.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "Grain_ID", 
            "Mean_B", "Mean_G", "Mean_R", 
            "Mean_H", "Mean_S", "Mean_V", 
            "Mean_L", "Mean_A", "Mean_B_Lab",
            "Area", "Perimeter", "Circularity", "AspectRatio",
            "Classification", "Reasons"
        ])
    
        # We need the contours again to draw them correctly on the original image
        # Since segment_grains returns masks, let's re-find contours on the masks or modify segment_grains to return contours/bboxes.
        # For simplicity, let's just re-run the segmentation logic here or better, modify segment_grains to return more info.
        # Actually, segment_grains returns cropped images, so we lost the original position.
        # Let's refactor segment_grains slightly to return bounding boxes as well.
        # Wait, I can't easily change the signature without breaking other things (though I am the only user).
        # Let's just re-do the contour finding on the `opening` mask returned by segment_grains.
        
        _, opening = segment_grains(processed_image) # Re-run to get the mask
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours exactly as in segment_grains
        valid_contours = [c for c in contours if cv2.contourArea(c) >= 100]
        
        # Ensure we have the same number of contours as results
        if len(valid_contours) != len(results):
            print("Warning: Mismatch between contours and results during visualization.")
        
        for i, (contour, result, feats) in enumerate(zip(valid_contours, results, all_features)):
            x, y, w, h = cv2.boundingRect(contour)
            
            is_good = result['classification'] == "Good"
            color = (0, 255, 0) if is_good else (0, 0, 255) # Green vs Red
            
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            
            label = "Good" if is_good else "Bad"
            cv2.putText(result_image, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Write to CSV
            row = [i] + list(feats) + [result['classification'], "; ".join(result['reasons'])]
            writer.writerow(row)
            
    output_path = "result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Result image saved to {output_path}")
    print("Features saved to features.csv")

if __name__ == "__main__":
    main()
