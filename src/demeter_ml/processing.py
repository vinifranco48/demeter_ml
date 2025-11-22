import cv2
import numpy as np

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

def segment_grains(image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Segments grains from the background using thresholding and contour detection.
    Returns a list of masked grain images and the binary mask.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise and separate touching grains
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    grain_masks = []
    for contour in contours:
        # Filter small noise
        if cv2.contourArea(contour) < 100:
            continue
            
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Extract the grain using the mask
        grain = cv2.bitwise_and(image, image, mask=mask)
        
        # Crop to bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        cropped_grain = grain[y:y+h, x:x+w]
        grain_masks.append(cropped_grain)
        
    return grain_masks, opening
