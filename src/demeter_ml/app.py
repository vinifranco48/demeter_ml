import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory (src) to sys.path so we can import demeter_ml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demeter_ml.grain_classifier import GrainClassifierPipeline, GrainQuality
from demeter_ml.corn_classifier import CornClassifierPipeline, CornQuality
from demeter_ml.llm import analyze_grains_with_llm, identify_grain_type

st.set_page_config(page_title="Demeter ML - Grain Analysis", layout="wide")

st.title("🌾 Demeter ML: Grain Classification")
st.markdown("Upload an image of soybean or corn grains to analyze their quality.")
st.markdown("**Pipeline**: Classical Computer Vision + AI Report Generation")

# Sidebar
st.sidebar.header("Settings")

# LLM Settings
use_llm = st.sidebar.checkbox("Generate AI Report", value=True, help="Generate a summary report using Groq LLM")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

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
            st.image(image, channels="BGR", use_column_width=True)

        # Identify Grain Type
        grain_type = "Soybean"
        if GROQ_API_KEY:
            with st.spinner('Identifying grain type...'):
                grain_type = identify_grain_type(image, GROQ_API_KEY)
            st.success(f"Detected Grain Type: **{grain_type}**")
        else:
            st.info("Grain type detection disabled (no API key). Defaulting to Soybean.")

        with st.spinner(f'Processing {grain_type} with specialized pipeline...'):
            # Select appropriate pipeline
            if grain_type == "Corn":
                pipeline = CornClassifierPipeline()
                QualityEnum = CornQuality
            else:
                pipeline = GrainClassifierPipeline()
                QualityEnum = GrainQuality
                
            annotated_image, report = pipeline.process(image)

            if report['total_graos'] == 0:
                st.warning("No grains found in the image.")
            else:
                # Display Result Image
                with col2:
                    st.subheader("Analyzed Image")
                    st.image(annotated_image, channels="BGR", use_column_width=True)

                # Metrics
                total_grains = report['total_graos']
                
                # Calculate counts from report (handling potentially different enum values)
                # CornQuality: TIPO_1, TIPO_2, TIPO_3, FORA_TIPO, DESCARTE
                # GrainQuality: EXCELENTE, BOM, REGULAR, DEFEITUOSO, DANIFICADO
                
                if grain_type == "Corn":
                    good_counts = (report['classificacao'].get(CornQuality.TIPO_1.value, 0) + 
                                 report['classificacao'].get(CornQuality.TIPO_2.value, 0))
                    regular_counts = report['classificacao'].get(CornQuality.TIPO_3.value, 0)
                    bad_counts = (report['classificacao'].get(CornQuality.FORA_TIPO.value, 0) + 
                                report['classificacao'].get(CornQuality.DESCARTE.value, 0))
                else:
                    good_counts = (report['classificacao'].get(GrainQuality.EXCELENTE.value, 0) + 
                                 report['classificacao'].get(GrainQuality.BOM.value, 0))
                    regular_counts = report['classificacao'].get(GrainQuality.REGULAR.value, 0)
                    bad_counts = (report['classificacao'].get(GrainQuality.DEFEITUOSO.value, 0) + 
                                report['classificacao'].get(GrainQuality.DANIFICADO.value, 0))

                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Grains", total_grains)
                m2.metric("Good/Type 1-2", good_counts)
                m3.metric("Regular/Type 3", regular_counts)
                m4.metric("Defective/Out of Type", bad_counts, delta_color="inverse")

                # Detailed Data
                st.subheader("Detailed Analysis")

                # Create DataFrame
                data = []
                for grain in report['graos']:
                    row = {
                        "ID": grain['id'],
                        "Quality": grain['qualidade'],
                        "Confidence": f"{grain['confianca']:.2f}",
                        "Defects": ", ".join(grain['defeitos']) if grain['defeitos'] else "None",
                        "Area": grain['area'],
                        "Circularity": grain['circularidade'],
                        "Aspect Ratio": grain['aspect_ratio']
                    }
                    data.append(row)

                df = pd.DataFrame(data)
                st.dataframe(df)
                
                # Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Data as CSV",
                    csv,
                    "grain_analysis.csv",
                    "text/csv",
                    key='download-csv'
                )

                # AI Report Section
                if use_llm and GROQ_API_KEY:
                    st.divider()
                    st.subheader("🤖 AI Agronomist Report")
                    
                    with st.spinner("Generating AI analysis report..."):
                        # Prepare data for LLM
                        summary_stats = {
                            "total": total_grains,
                            "good": good_counts,
                            "bad": bad_counts + regular_counts, 
                            "averages": report['estatisticas'],
                            "grain_type": grain_type
                        }
                        
                        # Prepare detailed results for LLM
                        detailed_results = []
                        for grain in report['graos']:
                            # Map quality to simple classification
                            if grain_type == "Corn":
                                is_good = grain['qualidade'] in [CornQuality.TIPO_1.value, CornQuality.TIPO_2.value]
                            else:
                                is_good = grain['qualidade'] in [GrainQuality.EXCELENTE.value, GrainQuality.BOM.value]
                                
                            classification = "Good" if is_good else "Defect"
                            
                            detailed_results.append({
                                "classification": classification,
                                "reasons": grain['defeitos']
                            })
                            
                        # Call LLM function
                        try:
                            llm_report = analyze_grains_with_llm(
                                summary_stats=summary_stats,
                                detailed_results=detailed_results,
                                api_key=GROQ_API_KEY,
                                image=annotated_image
                            )
                            st.markdown(llm_report)
                        except Exception as e:
                            st.error(f"Error generating AI report: {e}")
