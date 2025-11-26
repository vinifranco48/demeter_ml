import cv2
import numpy as np
from typing import Optional

def load_image(path: str) -> np.ndarray:
    """
    Loads an image from the specified path.
    """
    # Use numpy to read the file into a buffer, then decode it
    # This handles paths with non-ASCII characters on Windows
    stream = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    
    if image is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Applies Median blur to reduce noise while preserving edges.
    """
    # Median blur is better for salt-and-pepper noise and preserving edges
    blurred = cv2.medianBlur(image, 5)
    return blurred

def sharpen_image(image: np.ndarray) -> np.ndarray:
    """
    Sharpen the image using a Laplacian filter to enhance edges.
    """
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def segment_grains(image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, list, np.ndarray]:
    """
    Segments grains using Watershed algorithm to separate touching grains.
    Returns:
        - grain_masks: List of cropped grain images
        - markers: The watershed markers (labeled image)
        - contours: List of contours found
        - hierarchy: Contour hierarchy
    """
    # 1. Convert to Gray and Sharpen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Thresholding (Otsu)
    # Use binary inverse because grains are usually light on dark or we want the object to be white
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 4. Sure background area (Dilate)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 5. Sure foreground area (Distance Transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    
    # 6. Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 7. Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # 8. Watershed
    markers = cv2.watershed(image, markers)
    
    # 9. Extract individual grains based on markers
    grain_masks = []
    unique_markers = np.unique(markers)
    
    # Re-find contours on the original opening mask to get hierarchy for holes
    # We use RETR_CCOMP to get a 2-level hierarchy (External + Holes)
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Map watershed markers to contours is tricky. 
    # For simplicity in this pipeline, we will use the contours from `opening` for feature extraction
    # but use Watershed to visually separate them if we were doing pixel-wise analysis.
    # HOWEVER, the user specifically asked for Watershed to separate touching grains.
    # If we just use `findContours` on `opening`, we lose the watershed separation if they are touching.
    
    # Let's rely on the Watershed markers to generate the masks for each grain.
    # We will iterate over each unique marker (skipping background and boundaries)
    
    final_contours = []
    final_hierarchy = [] # We might lose hierarchy if we split touching grains, but let's try to preserve what we can.
    
    # Actually, if we use Watershed, we get labeled regions. We can create a mask for each label.
    for marker in unique_markers:
        if marker == 0 or marker == 1 or marker == -1: # 0: Unknown, 1: Background, -1: Boundary
            continue
            
        # Create a mask for this specific grain
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers == marker] = 255
        
        # Find contour of this specific grain mask to get bounding rect and shape
        # We use RETR_EXTERNAL here because we isolated the grain
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
            
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 100:
            continue
            
        final_contours.append(c)
        
        # Extract the grain
        grain = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(c)
        cropped_grain = grain[y:y+h, x:x+w]
        grain_masks.append(cropped_grain)
        
    # Note: By doing this per-marker extraction, we lose the "Hole" hierarchy from the original full image scan
    # if we don't re-scan the individual grain mask for holes.
    # To support holes, we should check for internal contours WITHIN the specific grain mask.
    
    # Let's reconstruct a hierarchy list that matches `grain_masks`.
    # Each entry will be a list of hole contours for that grain.
    grain_holes = []
    for marker in unique_markers:
        if marker <= 1: continue
        
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers == marker] = 255
        
        # Find contours with hierarchy on this isolated mask
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts: continue
        
        # The largest contour is the grain boundary (Parent)
        # Any other contours are holes (Children)
        # hierarchy format: [Next, Previous, First_Child, Parent]
        
        # We need to find the index of the external contour
        # Usually it's the one with Parent = -1
        
        if hier is None:
            grain_holes.append([])
            continue
            
        holes = []
        # hier[0] is the array of hierarchies
        for i, h_info in enumerate(hier[0]):
            # h_info = [Next, Previous, First_Child, Parent]
            parent_idx = h_info[3]
            if parent_idx != -1: # It has a parent, so it's a hole
                holes.append(cnts[i])
                
        grain_holes.append(holes)

    return grain_masks, markers, final_contours, grain_holes


def analyze_grains_integrated(
    image: np.ndarray,
    api_key: Optional[str] = None,
    use_llm_for_uncertain: bool = False
) -> tuple[list[dict], list[np.ndarray], list, list]:
    """
    Integrated pipeline that processes an image and analyzes grains using:
    1. Classical CV processing (preprocessing, segmentation, feature extraction)
    2. Rule-based analysis (fast, always runs first)
    3. LLM analysis for ALL grains (if api_key provided) or only uncertain cases

    Args:
        image: Input image (BGR format)
        api_key: Groq API key for LLM analysis (optional)
        use_llm_for_uncertain: If True, use LLM only for uncertain cases. If False and api_key provided, use for ALL grains.

    Returns:
        - results: List of analysis dictionaries for each grain
        - grain_masks: List of cropped grain images
        - contours: List of grain contours (global coordinates)
        - grain_holes: List of hole contours for each grain
    """
    from .features import extract_all_features
    from .analysis import analyze_grain_rules

    # Step 1: Preprocess
    print("Step 1/4: Preprocessing image...")
    processed_image = preprocess_image(image)
    sharpened_image = sharpen_image(processed_image)

    # Step 2: Segment grains
    print("Step 2/4: Segmenting grains with Watershed...")
    grain_masks, markers, contours, grain_holes = segment_grains(sharpened_image)

    if len(grain_masks) == 0:
        return [], [], [], []

    print(f"Step 3/4: Extracting features and analyzing {len(grain_masks)} grains...")

    results = []
    llm_used_count = 0
    use_llm_for_all = api_key is not None and not use_llm_for_uncertain

    # Step 3: Analyze each grain
    for i, (grain, contour, holes) in enumerate(zip(grain_masks, contours, grain_holes)):
        # Extract features
        features = extract_all_features(grain, contour=None)

        # Rule-based analysis (always run first for baseline)
        rule_analysis = analyze_grain_rules(features, holes=holes)

        # Determine if we should use LLM for this grain
        should_use_llm = False
        if api_key is not None:
            if use_llm_for_all:
                # Use LLM for ALL grains
                should_use_llm = True
            elif use_llm_for_uncertain and rule_analysis.get('uncertain', False):
                # Use LLM only for uncertain cases
                should_use_llm = True

        if should_use_llm:
            print(f"  Grain {i+1}/{len(grain_masks)}: Using LLM analysis...")

            try:
                # Import here to avoid circular dependency
                from .llm import analyze_single_grain_with_llm

                # Use LLM for enhanced analysis
                llm_result = analyze_single_grain_with_llm(
                    grain_image=grain,
                    features=features,
                    rule_analysis=rule_analysis,
                    api_key=api_key
                )

                # HYBRID DECISION LOGIC: Balance rules + LLM
                # Priority: Rules are the baseline, LLM validates/corrects

                rule_class = rule_analysis['classification']
                llm_class = llm_result['classification']
                llm_confidence = llm_result.get('confidence', 'medium')

                # PRIORITY DECISION LOGIC
                # Strong priority to rules baseline, LLM only refines/validates

                if rule_class == llm_class:
                    # Perfect agreement: use LLM classification with high confidence
                    final_classification = llm_class
                    final_reasons = llm_result['reasons']
                    final_confidence = "high"

                elif rule_class == "Good":
                    # Rules say Good - be VERY conservative about overriding
                    if "Defect" in llm_class and llm_confidence == "high":
                        # LLM detected a critical defect with high confidence
                        # Check if it's a severe defect type that rules might miss
                        critical_defects = ["Broken", "Perforated", "Rotten", "Insect"]
                        is_critical = any(defect in llm_class for defect in critical_defects)

                        if is_critical:
                            # Trust LLM for critical defects
                            final_classification = llm_class
                            final_reasons = llm_result['reasons'] + ["LLM detected critical visual defect"]
                            final_confidence = "medium"
                        else:
                            # Non-critical defect, trust rules (likely false positive)
                            final_classification = rule_class
                            final_reasons = ["Grão adequado por análise de features"] + [f"LLM sugeriu: {llm_class} (ignorado)"]
                            final_confidence = "high"
                    else:
                        # LLM not high confidence or agrees with Good -> trust rules
                        final_classification = rule_class
                        final_reasons = ["Grão adequado por análise de features"]
                        if llm_result['reasons']:
                            final_reasons.extend(llm_result['reasons'][:2])
                        final_confidence = "high"

                elif "Defect" in rule_class:
                    # Rules say Defect
                    if llm_class == "Good":
                        # LLM says good but rules found defect
                        if rule_analysis.get('uncertain', False) and llm_confidence == "high":
                            # Rules were uncertain, trust high-confidence LLM
                            final_classification = llm_class
                            final_reasons = llm_result['reasons'] + ["LLM corrigiu falso positivo das regras"]
                            final_confidence = "medium"
                        else:
                            # Rules confident about defect, be conservative
                            final_classification = rule_class
                            final_reasons = rule_analysis['reasons'] + ["Confirmado por análise de features"]
                            final_confidence = "high"
                    else:
                        # Both say defect, possibly different types
                        # Use LLM's more specific classification
                        final_classification = llm_class
                        final_reasons = llm_result['reasons']
                        final_confidence = "high"
                else:
                    # Fallback
                    final_classification = rule_class
                    final_reasons = rule_analysis['reasons']
                    final_confidence = "medium"

                final_result = {
                    "classification": final_classification,
                    "reasons": final_reasons,
                    "confidence": final_confidence,
                    "llm_used": llm_result['llm_used'],
                    "llm_verdict": llm_result.get('llm_verdict', ''),
                    "llm_reasoning": llm_result.get('llm_reasoning', ''),
                    "llm_classification": llm_class,
                    "llm_confidence": llm_confidence,
                    "rule_classification": rule_class,
                    "rule_reasons": rule_analysis['reasons'],
                    "uncertain": rule_analysis.get('uncertain', False),
                    "features": rule_analysis['features']
                }

                if llm_result['llm_used']:
                    llm_used_count += 1
            except Exception as e:
                print(f"  Warning: LLM analysis failed for grain {i}: {str(e)[:100]}")
                # Fallback to rule-based
                final_result = {
                    "classification": rule_analysis['classification'],
                    "reasons": rule_analysis['reasons'] + [f"LLM error: {str(e)[:50]}"],
                    "confidence": "medium",
                    "llm_used": False,
                    "uncertain": rule_analysis.get('uncertain', False),
                    "features": rule_analysis['features']
                }
        else:
            # Use rule-based analysis only
            final_result = {
                "classification": rule_analysis['classification'],
                "reasons": rule_analysis['reasons'],
                "confidence": "high" if not rule_analysis.get('uncertain', False) else "medium",
                "llm_used": False,
                "uncertain": rule_analysis.get('uncertain', False),
                "features": rule_analysis['features']
            }

        results.append(final_result)

    # Print summary
    print(f"Step 4/4: Analysis complete!")
    if llm_used_count > 0:
        print(f"  LLM analysis used for {llm_used_count}/{len(grain_masks)} grains.")

    return results, grain_masks, contours, grain_holes
