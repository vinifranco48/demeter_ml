import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory (src) to sys.path so we can import demeter_ml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demeter_ml.processing import preprocess_image, segment_grains
from demeter_ml.features import extract_all_features
from demeter_ml.analysis import analyze_grain_rules

st.set_page_config(page_title="Demeter ML - Grain Analysis", layout="wide")

st.title("🌾 Demeter ML: Grain Classification")
st.markdown("Upload an image of soybean or corn grains to analyze their quality.")

# Sidebar
st.sidebar.header("Settings")
# Future: Add threshold sliders here
# circularity_threshold = st.sidebar.slider("Circularity Threshold", 0.0, 1.0, 0.7)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is None:
        st.error("Error loading image.")
    else:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, channels="BGR", use_container_width=True)
            
        with st.spinner('Processing...'):
            # Pipeline
            processed_image = preprocess_image(image)
            grain_masks, opening = segment_grains(processed_image)
            
            if not grain_masks:
                st.warning("No grains found in the image.")
            else:
                # Analysis
                results = []
                all_features = []
                
                # For visualization
                result_image = image.copy()
                
                # Re-find contours for visualization (logic copied from main.py)
                contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_contours = [c for c in contours if cv2.contourArea(c) >= 100]
                
                # Sort contours/masks to match? 
                # segment_grains returns masks in order of contours found.
                # But we need to be careful. 
                # Let's trust the order is preserved or re-extract features from valid_contours directly to be safe?
                # Actually, segment_grains logic: finds contours, filters < 100, then extracts.
                # So valid_contours should match grain_masks 1:1.
                
                for i, (grain, contour) in enumerate(zip(grain_masks, valid_contours)):
                    feats = extract_all_features(grain)
                    analysis = analyze_grain_rules(feats)
                    
                    results.append(analysis)
                    all_features.append(feats)
                    
                    # Draw on result image
                    x, y, w, h = cv2.boundingRect(contour)
                    is_good = analysis['classification'] == "Good"
                    color = (0, 255, 0) if is_good else (0, 0, 255)
                    
                    cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(result_image, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Display Result Image
                with col2:
                    st.subheader("Analyzed Image")
                    st.image(result_image, channels="BGR", use_container_width=True)
                
                # Metrics
                total_grains = len(results)
                good_grains = sum(1 for r in results if r['classification'] == "Good")
                bad_grains = total_grains - good_grains
                
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Grains", total_grains)
                m2.metric("Good Grains", good_grains)
                m3.metric("Defective Grains", bad_grains, delta_color="inverse")
                
                # Detailed Data
                st.subheader("Detailed Analysis")
                
                # Create DataFrame
                data = []
                for i, (res, feats) in enumerate(zip(results, all_features)):
                    row = {
                        "ID": i,
                        "Classification": res['classification'],
                        "Reasons": ", ".join(res['reasons']),
                        "Circularity": f"{feats[11]:.2f}",
                        "Aspect Ratio": f"{feats[12]:.2f}",
                        "Mean V (Brightness)": f"{feats[5]:.1f}",
                        "Mean H (Hue)": f"{feats[3]:.1f}"
                    }
                    data.append(row)
                
                df = pd.DataFrame(data)
                
                # Style the dataframe
                def highlight_row(row):
                    return ['background-color: #ffcdd2' if row['Classification'] != 'Good' else 'background-color: #c8e6c9'] * len(row)

                st.dataframe(df, use_container_width=True)
                
                # Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Data as CSV",
                    csv,
                    "grain_analysis.csv",
                    "text/csv",
                    key='download-csv'
                )
