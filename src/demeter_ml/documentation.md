# Documentação do Código `demeter_ml`

Este documento fornece uma explicação detalhada da estrutura e funcionalidade do pacote `demeter_ml`, responsável pela análise e classificação de grãos (Soja e Milho) utilizando Visão Computacional Clássica e Inteligência Artificial Generativa (LLM).

## 1. Visão Geral

O projeto tem como objetivo automatizar a classificação de grãos agrícolas, identificando defeitos e qualidade comercial. Ele utiliza uma abordagem híbrida:
1.  **Visão Computacional (OpenCV)**: Para segmentação, medição e extração de características visuais (cor, forma, textura).
2.  **Regras Especialistas**: Para classificação rápida e determinística baseada em normas técnicas.
3.  **LLM (Llama via Groq)**: Para análise contextual, identificação do tipo de grão e validação de casos incertos.

## 2. Estrutura do Projeto

Os arquivos principais estão localizados em `src/demeter_ml/`:

*   **`app.py`**: Interface do usuário (Streamlit).
*   **`processing.py`**: Núcleo de processamento de imagem (segmentação e orquestração).
*   **`grain_classifier.py`**: Pipeline específica para **Soja**.
*   **`corn_classifier.py`**: Pipeline específica para **Milho**.
*   **`llm.py`**: Integração com modelos de linguagem (Groq API).
*   **`features.py`**: Extração de características matemáticas das imagens.
*   **`analysis.py`**: Lógica de decisão baseada em regras e detecção de incerteza.
*   **`clustering.py`**: Utilitário para agrupamento não supervisionado (K-Means).

---

## 3. Detalhamento dos Módulos

### 3.1 Interface do Usuário (`app.py`)
Este é o ponto de entrada da aplicação web.
*   **Funcionalidade**:
    *   Permite upload de imagens.
    *   Identifica automaticamente se é Milho ou Soja usando LLM (`identify_grain_type`).
    *   Seleciona a pipeline correta (`CornClassifierPipeline` ou `GrainClassifierPipeline`).
    *   Exibe a imagem original e a processada (segmentada).
    *   Mostra métricas (contagem, qualidade) e permite download dos dados em CSV.
    *   Gera um relatório agronômico detalhado usando IA (`analyze_grains_with_llm`).

### 3.2 Processamento de Imagem (`processing.py`)
Contém as funções fundamentais de manipulação de imagem.
*   **`preprocess_image`**: Aplica desfoque mediano para reduzir ruído preservando bordas.
*   **`sharpen_image`**: Realça bordas para facilitar a segmentação.
*   **`segment_grains`**: A função mais complexa. Utiliza o algoritmo **Watershed** para separar grãos que estão se tocando.
    *   Passos: Escala de cinza -> Threshold (Otsu) -> Morfologia -> Distance Transform -> Watershed.
*   **`analyze_grains_integrated`**: Orquestrador que combina a extração de features, análise de regras e, se necessário, consulta o LLM para refinar a classificação de cada grão.

```python
# Exemplo de Segmentação com Watershed (processing.py)
def segment_grains(image: np.ndarray):
    # 1. Escala de cinza e Threshold (Otsu)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Remoção de ruído e definição de fundo/frente
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 3. Distance Transform para encontrar centros dos grãos
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    
    # 4. Watershed
    _, markers = cv2.connectedComponents(np.uint8(sure_fg))
    markers = markers + 1
    markers[cv2.subtract(sure_bg, np.uint8(sure_fg)) == 255] = 0
    markers = cv2.watershed(image, markers)
    
    return markers # Retorna marcadores para separar os grãos
```

### 3.3 Classificadores Específicos

#### Soja (`grain_classifier.py`)
*   Define classes de qualidade (`EXCELENTE`, `BOM`, `REGULAR`, `DEFEITUOSO`, `DANIFICADO`).
*   **`GrainClassifierPipeline`**:
    *   Configurada para detectar cores bege/amarelo claro.
    *   Regras focadas em circularidade (soja é redonda) e manchas escuras.

```python
# Exemplo de Classificação de Soja (grain_classifier.py)
def classify(self, grains: List[GrainFeatures]) -> List[GrainFeatures]:
    for grain in grains:
        # Detecta defeitos baseados em regras
        defects = self._detect_defects(grain)
        grain.defects = defects
        
        # Calcula qualidade final
        quality, confidence = self._calculate_quality(grain, defects)
        grain.quality = quality
        grain.confidence = confidence
    return grains
```

#### Milho (`corn_classifier.py`)
*   Define classes conforme norma brasileira (`TIPO_1`, `TIPO_2`, `TIPO_3`, `FORA_TIPO`).
*   **`CornClassifierPipeline`**:
    *   Otimizada para tons de amarelo intenso e laranja.
    *   Detecta defeitos específicos do milho: **Ardido** (fermentado/escuro), **Mofado**, **Quebrado**, **Choco**.
    *   Usa índices específicos como `yellow_index` para avaliar a cor.

### 3.4 Inteligência Artificial (`llm.py`)
Módulo de comunicação com a API da Groq (usando modelos Llama Vision).
*   **`identify_grain_type`**: Envia uma versão reduzida da imagem para o LLM classificar entre "Milho" ou "Soja".
*   **`analyze_single_grain_with_llm`**: Analisa um único grão recortado quando a análise por regras é incerta. O prompt instrui o modelo a agir como um agrônomo.
*   **`analyze_grains_with_llm`**: Gera o relatório final textual, analisando as estatísticas gerais e a imagem completa anotada.

```python
# Prompt para o LLM (llm.py)
def analyze_single_grain_with_llm(grain_image, features, rule_analysis, api_key):
    # ... (código de codificação da imagem) ...
    
    text_prompt = f"""
    Você é um AGRÔNOMO especialista em classificação de grãos.
    
    Dados do grão:
    - Classificação preliminar (Regras): {rule_analysis['classification']}
    - Razões: {', '.join(rule_analysis['reasons'])}
    - Circularidade: {features[11]:.2f} (1.0 é perfeito)
    
    Analise a imagem e confirme se é 'Good' ou 'Defect'.
    Responda em JSON.
    """
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[...], # Envia prompt + imagem
        temperature=0.3
    )
    return json.loads(response.choices[0].message.content)
```

### 3.5 Análise e Features (`analysis.py` e `features.py`)
A "matemática" por trás da classificação.
*   **`features.py`**:
    *   **Cor**: Médias em RGB, HSV e LAB.
    *   **Forma**: Área, Perímetro, Circularidade, Aspect Ratio (largura/altura), Solidez.
    *   **Textura**: Contraste e Entropia (para detectar grãos enrugados ou mofados).
*   **`analysis.py`**:
    *   **`analyze_grain_rules`**: Aplica limiares rígidos (ex: se `circularidade < 0.7`, então é defeituoso).
    *   **`is_uncertain_case`**: Detecta se um grão está "na borda" dos limiares, sugerindo que o LLM deve ser consultado para uma segunda opinião.

```python
# Extração de Features (features.py)
def extract_all_features(grain_image, contour=None):
    # Cor (Médias BGR, HSV, LAB)
    color_feats = extract_color_features(grain_image)
    
    # Forma (Área, Perímetro, Circularidade, etc.)
    shape_feats = extract_shape_features(contour)
    
    # Textura (LBP, Contraste, Entropia)
    texture_feats = extract_texture_features(grain_image)
    
    return np.concatenate([color_feats, shape_feats, texture_feats])
```

---

## 4. Fluxo de Execução

1.  **Upload**: O usuário envia uma imagem no `app.py`.
2.  **Identificação**: O sistema decide se é Milho ou Soja.
3.  **Segmentação (`processing.py`)**: A imagem é tratada e os grãos são separados do fundo e uns dos outros.
4.  **Extração (`features.py`)**: Cada grão é recortado e medido (cor, tamanho, forma).
5.  **Classificação Híbrida**:
    *   Primeiro, as **Regras** (`analysis.py` / `*_classifier.py`) dão um veredito inicial.
    *   Se o veredito for incerto (ex: grão levemente oval), o **LLM** (`llm.py`) analisa a imagem do grão.
    *   O sistema decide a classificação final com base na confiança de ambos.
6.  **Relatório**: Os resultados são agregados, exibidos na tela e um relatório final em texto é gerado.
