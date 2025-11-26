# Lógica Híbrida: Regras + LLM

## 🎯 Problema Resolvido

**Problema Original**: O LLM estava classificando grãos saudáveis como defeituosos, sendo excessivamente conservador.

**Solução**: Implementar uma lógica de decisão híbrida que combina **análise por regras** (baseline confiável) + **LLM com visão** (detecta casos adversos) de forma equilibrada.

## 🧠 Como Funciona

### 1. Análise Sequencial

```
Imagem → Processamento CV → Segmentação → Extração de Features
                                                    ↓
                                    [1] Análise por Regras (sempre)
                                                    ↓
                                    [2] Análise LLM (se habilitado)
                                                    ↓
                                    [3] Lógica de Decisão Híbrida
                                                    ↓
                                         Classificação Final
```

### 2. Lógica de Decisão Híbrida

A decisão final depende da concordância/discordância entre regras e LLM:

#### ✅ Caso 1: Concordância
```
Regras: Good  ┐
              ├─→ FINAL: Good (confiança: high)
LLM: Good     ┘
```
**Decisão**: Usa classificação do LLM com confiança alta.

#### ⚠️ Caso 2: Regras=Good, LLM=Defect
```
Regras: Good                LLM Confiança: High
LLM: Defect (ex: mancha)    ↓
                           FINAL: Defect (confiança: medium)
                           Motivo: "LLM detected visual defect"

Regras: Good                LLM Confiança: Low/Medium
LLM: Defect                 ↓
                           FINAL: Good (confiança: medium)
                           Motivo: "LLM uncertain: Defect..."
```
**Lógica**:
- Se LLM tem **alta confiança**, pode ter detectado defeito visual que as regras perderam → **Confia no LLM**
- Se LLM tem **baixa/média confiança**, as regras são baseline confiável → **Confia nas Regras**

#### ⚠️ Caso 3: Regras=Defect, LLM=Good
```
Regras: Defect (incerto)    LLM Confiança: High
LLM: Good                   ↓
                           FINAL: Good (confiança: medium)
                           Motivo: "LLM corrected false positive"

Regras: Defect (certo)      LLM Confiança: Low/Medium
LLM: Good                   ↓
                           FINAL: Defect (confiança: medium)
                           Motivo: "LLM disagreed but rules confirmed"
```
**Lógica**:
- Se regras eram **incertas** e LLM tem **alta confiança**, pode ser falso positivo → **Confia no LLM**
- Se regras eram **certas**, mantém postura conservadora → **Confia nas Regras**

#### 🔀 Caso 4: Tipos de Defeito Diferentes
```
Regras: Defect: Broken
LLM: Defect: Rotten
↓
FINAL: Defect: Rotten (confiança: low)
Motivo: ["Observações LLM...", "Rule-based: Broken"]
```
**Lógica**: Usa classificação do LLM mas marca confiança baixa pela discordância.

## 📊 Matriz de Decisão

| Regras | LLM | Confiança LLM | Regras Incertas? | **FINAL** | Confiança |
|--------|-----|---------------|------------------|-----------|-----------|
| Good | Good | Qualquer | Qualquer | **Good** | High |
| Good | Defect | High | Não importa | **Defect** | Medium |
| Good | Defect | Low/Med | Não importa | **Good** | Medium |
| Defect | Good | High | Sim | **Good** | Medium |
| Defect | Good | High | Não | **Defect** | Medium |
| Defect | Good | Low/Med | Qualquer | **Defect** | Medium |
| Defect:A | Defect:B | Qualquer | Qualquer | **Defect:B** | Low |

## 🔧 Melhorias no Prompt do LLM

### Antes (Muito Rigoroso)
```
"Analise este grão e determine se está ADEQUADO ou DEFEITUOSO.
Considere: manchas, rachaduras, deformações..."
```
❌ Problema: Classificava variações naturais como defeitos.

### Depois (Realista e Balanceado)
```
"IMPORTANTE: Seja REALISTA e não excessivamente rigoroso.
Pequenas imperfeições naturais são NORMAIS e aceitáveis.

✅ ADEQUADO se: forma regular, cor uniforme, sem danos graves
❌ DEFEITUOSO APENAS se: rachaduras VISÍVEIS, furos claros,
   manchas EXTENSAS, deformação SEVERA

NÃO classifique como defeituoso por: pequenas variações de cor,
forma levemente oval, textura natural, sombras da iluminação

Priorize o que você VÊ na imagem."
```
✅ Resultado: Mais preciso e menos falsos positivos.

## 🎨 Indicadores Visuais

### Na Interface Web (Streamlit)
- ⚡ = Discordância entre regras e LLM
- [LLM] = Grão analisado pelo LLM
- ⚠ = Caso incerto detectado
- Cores: Verde (Good), Vermelho (Defect), Laranja (Incerto sem LLM)

### No Terminal (CLI)
```
Grain 5 [LLM] ⚡: Good (confidence: medium)
  Reasons: ['Forma adequada', 'LLM uncertain: Defect...']
  LLM Classification: Defect: Irregular (confidence: low)
  LLM Verdict: defeituoso
  Rule Classification: Good
  ⚡ DISAGREEMENT: Rules said 'Good', LLM said 'Defect: Irregular'. Final: 'Good'
```

## 📈 Benefícios da Abordagem Híbrida

1. **Reduz Falsos Positivos**: Não marca grãos bons como ruins
2. **Mantém Detecção de Defeitos**: LLM ainda detecta problemas visuais que regras perdem
3. **Transparência**: Usuário vê ambas as análises e a decisão final
4. **Confiança Calibrada**: Indica quando há incerteza/discordância
5. **Flexível**: Pode ajustar pesos dando prioridade a regras ou LLM

## 🔬 Casos de Teste

### Teste 1: Grão Perfeitamente Redondo
```
Regras: Good (circularidade=0.95, brilho=120)
LLM: Good (confidence=high, "grão saudável, cor uniforme")
→ FINAL: Good (confidence=high) ✅
```

### Teste 2: Grão Levemente Oval (Natural)
```
Regras: Good (circularidade=0.72, aspecto=0.78) [incerto]
LLM: Good (confidence=high, "forma oval natural")
→ FINAL: Good (confidence=high) ✅
```

### Teste 3: Grão com Rachadura Sutil
```
Regras: Good (todas features OK)
LLM: Defect: Broken (confidence=high, "rachadura visível no centro")
→ FINAL: Defect: Broken (confidence=medium) ✅
```

### Teste 4: Grão Irregular mas Saudável
```
Regras: Defect: Irregular (circularidade=0.65) [incerto]
LLM: Good (confidence=high, "apenas forma natural irregular")
→ FINAL: Good (confidence=medium) ✅
```

### Teste 5: Falso Positivo (Sombra)
```
Regras: Good
LLM: Defect: Dark (confidence=low, "área escura")
→ FINAL: Good (confidence=medium, "LLM uncertain") ✅
```

## 🛠️ Configuração

Para habilitar/desabilitar a lógica híbrida, use:

### CLI
```bash
# LLM para todos (híbrido ativo)
python -m demeter_ml.main imagem.jpg --llm-mode all

# LLM apenas incertos (híbrido seletivo)
python -m demeter_ml.main imagem.jpg --llm-mode uncertain

# Sem LLM (apenas regras)
python -m demeter_ml.main imagem.jpg --llm-mode none
```

### Streamlit
- Checkbox: "Enable LLM Analysis"
- Radio: "All grains" (híbrido em todos) ou "Uncertain cases only" (híbrido seletivo)

## 📝 Logs de Exemplo

```
Step 3/4: Extracting features and analyzing 10 grains...
  Grain 1/10: Using LLM analysis...
  Grain 2/10: Using LLM analysis...
Step 4/4: Analysis complete!
  LLM analysis used for 10/10 grains.

Summary: 8/10 grains are Good.
  - LLM used: 10 cases
  - Uncertain cases: 2
  - Disagreements: 3 (Rules vs LLM)
```

## 🔮 Melhorias Futuras

1. **Pesos Ajustáveis**: Permitir usuário configurar peso das regras vs LLM
2. **Aprendizado**: Coletar feedback para calibrar thresholds
3. **Ensemble**: Adicionar terceiro modelo (ex: CNN treinada)
4. **Explicabilidade**: Grad-CAM para mostrar áreas que LLM focou
5. **Métricas**: Calcular precisão/recall em dataset rotulado

## 📚 Referências

- `processing.py:242-313` - Lógica de decisão híbrida
- `llm.py:57-106` - Prompt melhorado do LLM
- `app.py:117-140` - Interface com indicadores de discordância
- `main.py:62-81` - CLI com informações de discordância
