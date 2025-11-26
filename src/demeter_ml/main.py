import argparse
import cv2
import numpy as np
import os
import json
import csv
from .grain_classifier import GrainClassifierPipeline, GrainQuality
from .llm import analyze_grains_with_llm

def main():
    parser = argparse.ArgumentParser(description="Grain Classification System")
    parser.add_argument("image_path", help="Path to the image to analyze")
    parser.add_argument("--api-key", help="Groq API key for LLM report (optional)", default=None)
    
    args = parser.parse_args()
    
    # Handle API key
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    
    # Load image
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
        
    # Read image using numpy to handle potential path encoding issues
    stream = np.fromfile(args.image_path, dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"Error: Could not decode image at {args.image_path}")
        return

    print("\nProcessing image with GrainClassifierPipeline...")
    
    # Initialize and run pipeline
    pipeline = GrainClassifierPipeline()
    annotated_image, report = pipeline.process(image)
    
    # Print basic results
    print(f"\nFound {report['total_graos']} grains.")
    print("\nClassification Summary:")
    for quality, count in report['classificacao'].items():
        if count > 0:
            print(f"  - {quality}: {count} ({report['percentuais'][quality]}%)")
            
    print("\nDefects Found:")
    if report['defeitos_encontrados']:
        for defect, count in report['defeitos_encontrados'].items():
            print(f"  - {defect}: {count}")
    else:
        print("  - None")

    # Save result image
    output_image_path = "result.jpg"
    cv2.imwrite(output_image_path, annotated_image)
    print(f"\nResult image saved to {output_image_path}")
    
    # Save CSV results
    csv_path = "analysis_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Grain_ID", "Quality", "Confidence", "Area", 
            "Circularity", "Aspect_Ratio", "Defects"
        ])
        
        for grain in report['graos']:
            writer.writerow([
                grain['id'],
                grain['qualidade'],
                grain['confianca'],
                grain['area'],
                grain['circularidade'],
                grain['aspect_ratio'],
                ", ".join(grain['defeitos'])
            ])
            
    print(f"Detailed results saved to {csv_path}")
    
    # Generate LLM Report if API key is available
    if api_key:
        print("\nGenerating AI Report...")
        
        # Prepare data for LLM
        # Map pipeline categories to simple Good/Bad for summary
        good_count = (report['classificacao'].get(GrainQuality.EXCELENTE.value, 0) + 
                     report['classificacao'].get(GrainQuality.BOM.value, 0))
        
        bad_count = report['total_graos'] - good_count
        
        summary_stats = {
            "total": report['total_graos'],
            "good": good_count,
            "bad": bad_count,
            "averages": report['estatisticas']
        }
        
        # Prepare detailed results format expected by LLM function
        detailed_results = []
        for grain in report['graos']:
            # Map quality to classification string
            classification = "Good" if grain['qualidade'] in [GrainQuality.EXCELENTE.value, GrainQuality.BOM.value] else "Defect"
            
            detailed_results.append({
                "classification": classification,
                "reasons": grain['defeitos']
            })
            
        try:
            llm_report = analyze_grains_with_llm(
                summary_stats=summary_stats,
                detailed_results=detailed_results,
                api_key=api_key,
                image=annotated_image
            )
            
            print("\n" + "="*50)
            print("AI ANALYSIS REPORT")
            print("="*50)
            print(llm_report)
            
            # Save report to text file
            with open("ai_report.txt", "w", encoding="utf-8") as f:
                f.write(llm_report)
            print("\nReport saved to ai_report.txt")
            
        except Exception as e:
            print(f"\nError generating LLM report: {e}")
    else:
        print("\nSkipping AI report (no API key provided)")

if __name__ == "__main__":
    main()
