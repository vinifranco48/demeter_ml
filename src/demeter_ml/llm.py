import os
import json
import numpy as np
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def get_groq_client(api_key: str = None):
    """
    Returns a configured OpenAI client for Groq.
    """
    if not OpenAI:
        return None
        
    # Use provided key or environment variable
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return None
        
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=key
    )

import cv2
import base64

def encode_image(image: np.ndarray) -> str:
    """Encodes an OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def analyze_single_grain_with_llm(grain_image: np.ndarray, features: np.ndarray, rule_analysis: dict, api_key: str) -> dict:
    """
    Analyzes a single grain using LLM vision model for cases where rule-based analysis is uncertain.
    Returns enhanced classification with LLM insights.
    """
    client = get_groq_client(api_key)
    if not client:
        return {
            "classification": rule_analysis['classification'],
            "reasons": rule_analysis['reasons'] + ["LLM não disponível"],
            "confidence": "low",
            "llm_used": False
        }

    # Prepare feature summary for context
    feature_summary = {
        "circularity": float(features[11]),
        "aspect_ratio": float(features[12]),
        "brightness": float(features[5]),
        "hue": float(features[3]),
        "entropy": float(features[17])
    }

    text_prompt = f"""
    Você é um especialista agrônomo em classificação comercial de grãos (soja/milho).

    **IMPORTANTE**: Seja REALISTA e TOLERANTE. A maioria dos grãos comerciais tem pequenas imperfeições naturais.
    Classifique como DEFEITUOSO apenas se houver dano REAL e VISÍVEL que afete a comercialização.

    📊 Dados técnicos (visão computacional):
    - Circularidade: {feature_summary['circularity']:.2f} (ideal: > 0.70)
    - Proporção aspecto: {feature_summary['aspect_ratio']:.2f} (ideal: 0.75-1.25)
    - Brilho: {feature_summary['brightness']:.1f} (ideal: > 60)
    - Matiz: {feature_summary['hue']:.1f} (verde imaduro: 35-85)

    🔍 Análise prévia (regras): {rule_analysis['classification']}
    Razões: {', '.join(rule_analysis['reasons']) if rule_analysis['reasons'] else 'Nenhum defeito detectado'}

    ⚠️ ATENÇÃO: Se a análise prévia disse "Good", você deve ter MUITA CERTEZA de que há um defeito grave
    antes de discordar. A análise por regras é geralmente confiável para grãos saudáveis.

    ---

    ✅ **Classifique como ADEQUADO (Good)** se o grão parece comercialmente aceitável:
    - Forma regular (arredondado/oval/levemente irregular é NORMAL)
    - Cor típica: amarelo/bege/marrom/dourado (variações naturais são OK)
    - Superfície: lisa ou textura natural do grão
    - Pode ter PEQUENAS imperfeições naturais (aceitável)

    ❌ **Classifique como DEFEITUOSO** SOMENTE se você VÊ claramente:
    - "Defect: Broken" → Rachadura GRANDE e visível, grão partido
    - "Defect: Perforated" → Furos EVIDENTES (ataque de insetos)
    - "Defect: Rotten" → Manchas ESCURAS/PRETAS extensas (podridão)
    - "Defect: Immature" → Cor VERDE predominante (imaturo)
    - "Defect: Damaged" → Deformação SEVERA ou dano estrutural grave

    🚫 **NÃO classifique como defeituoso por**:
    - Forma levemente oval/alongada (COMUM e aceitável!)
    - Pequenas variações de cor ou tonalidade
    - Textura superficial típica do grão
    - Sombras, reflexos, ou iluminação
    - Pequenas irregularidades naturais

    ---

    Responda em JSON (SEJA CONSERVADOR - se em dúvida, diga "Good"):
    {{
        "classification": "Good" OU "Defect: [tipo específico]",
        "confidence": "high", "medium" ou "low",
        "visual_observations": ["descreva o que VÊ"],
        "final_verdict": "adequado" ou "defeituoso",
        "reasoning": "por que tomou essa decisão"
    }}

    🎯 LEMBRE-SE: Analise VISUALMENTE a imagem. Se o grão PARECE comercialmente bom, classifique como "Good".
    """

    messages = [
        {"role": "system", "content": "Você é um especialista em classificação de grãos agrícolas. Responda APENAS com JSON válido."},
    ]

    base64_image = encode_image(grain_image)
    user_content = [
        {"type": "text", "text": text_prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent classification
            max_tokens=512,
            timeout=30.0  # 30 second timeout
        )

        # Parse JSON response
        llm_response = response.choices[0].message.content.strip()

        # Try to extract JSON from response
        if "```json" in llm_response:
            llm_response = llm_response.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_response:
            llm_response = llm_response.split("```")[1].split("```")[0].strip()

        result = json.loads(llm_response)

        return {
            "classification": result.get("classification", rule_analysis['classification']),
            "reasons": result.get("visual_observations", rule_analysis['reasons']),
            "confidence": result.get("confidence", "medium"),
            "llm_used": True,
            "llm_verdict": result.get("final_verdict", "unknown"),
            "llm_reasoning": result.get("reasoning", "")
        }

    except Exception as e:
        # Fallback to rule-based if LLM fails
        return {
            "classification": rule_analysis['classification'],
            "reasons": rule_analysis['reasons'] + [f"LLM erro: {str(e)[:50]}"],
            "confidence": "low",
            "llm_used": False
        }


def analyze_grains_with_llm(summary_stats: dict, detailed_results: list, api_key: str, image: np.ndarray = None) -> str:
    """
    Sends grain analysis data and optional image to Groq LLM for a detailed report.
    """
    client = get_groq_client(api_key)
    if not client:
        return "Erro: Biblioteca 'openai' não instalada ou chave API ausente."

    # Prepare a concise summary
    defects = [r for r in detailed_results if r['classification'] != "Good"]
    defect_summary = {}
    for d in defects:
        cls = d['classification']
        defect_summary[cls] = defect_summary.get(cls, 0) + 1

    prompt_data = {
        "total_grains": summary_stats['total'],
        "good_grains": summary_stats['good'],
        "defective_grains": summary_stats['bad'],
        "defect_types_count": defect_summary,
        "average_features": summary_stats.get('averages', {}),
        "sample_defects": [d['reasons'] for d in defects[:5]]
    }

    text_prompt = f"""
    Você é um especialista agrônomo em análise de qualidade de grãos (soja/milho).
    Analise os dados e a imagem fornecida (se houver) do sistema Demeter ML.

    Dados processados:
    {json.dumps(prompt_data, indent=2)}

    O modelo de visão pode ajudar a identificar padrões visuais, sujeira, ou detalhes que a visão clássica perdeu.
    Se a imagem contiver texto ou rótulos, use o OCR para incorporar essas informações.

    Forneça um relatório técnico detalhado em Português do Brasil (pt-BR):
    1. Resumo da qualidade do lote.
    2. Análise visual e dos dados sobre defeitos e causas.
    3. Recomendações agronômicas.
    """

    messages = [
        {"role": "system", "content": "Você é um assistente especialista em agronomia e visão computacional."},
    ]

    user_content = [{"type": "text", "text": text_prompt}]

    if image is not None:
        base64_image = encode_image(image)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct", # User requested model with OCR/Vision support
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na análise via LLM: {str(e)}"


def identify_grain_type(image: np.ndarray, api_key: str) -> str:
    """
    Identifies if the grain is Corn (Milho) or Soybean (Soja).
    Returns: "Corn" or "Soybean"
    """
    client = get_groq_client(api_key)
    if not client:
        return "Soybean" # Default

    # Resize for speed
    small_img = cv2.resize(image, (512, 512))
    base64_image = encode_image(small_img)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classifique esta imagem como 'Milho' ou 'Soja'. Responda APENAS com uma das duas palavras."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            temperature=0.1,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip().lower()
        
        if "milho" in content or "corn" in content:
            return "Corn"
        return "Soybean"
    except Exception:
        return "Soybean"
