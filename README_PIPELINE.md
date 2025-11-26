# Pipeline Integrada Demeter ML

## 🎯 Visão Geral

A pipeline integrada combina **Visão Computacional Clássica** + **LLM com OCR** para análise robusta de grãos em cenários adversos.

## 🏗️ Arquitetura

```
Imagem → Pré-processamento → Segmentação (Watershed) → Extração de Features
                                                          ↓
                                      Análise por Regras (rápida, baseline)
                                                          ↓
                                      [LLM com OCR] (para todos ou casos incertos)
                                                          ↓
                                            Classificação Final + Relatório
```

## 📊 Modos de Operação

### 1. **Modo "All grains"** (Padrão)
- LLM analisa **TODOS** os grãos
- Mais preciso, ideal para cenários adversos
- Usa OCR para detectar texto/sujeira
- Tempo: ~2-5s por grão

### 2. **Modo "Uncertain cases only"**
- LLM analisa apenas casos duvidosos (~10-20%)
- Mais rápido e econômico
- Casos incertos: features próximas aos thresholds

### 3. **Modo "None"** (Sem LLM)
- Apenas regras fixas (visão computacional clássica)
- Muito rápido
- Pode falhar em cenários adversos

## 🚀 Como Usar

### Via Interface Web (Streamlit)

```bash
streamlit run src/demeter_ml/app.py
```

**Configurações no Sidebar:**
- ✅ "Enable LLM Analysis": Ativa/desativa LLM
- 📊 "LLM Mode": Escolhe entre "All grains" ou "Uncertain cases only"

### Via Linha de Comando

```bash
# Análise de TODOS os grãos com LLM (padrão)
python -m demeter_ml.main imagem.jpg

# Análise apenas de casos incertos com LLM
python -m demeter_ml.main imagem.jpg --llm-mode uncertain

# Sem LLM (apenas regras)
python -m demeter_ml.main imagem.jpg --llm-mode none

# Com chave API específica
python -m demeter_ml.main imagem.jpg --api-key SUA_CHAVE_AQUI
```

## 🎨 Código de Cores nos Resultados

- 🟢 **Verde**: Grão adequado
- 🔴 **Vermelho**: Grão defeituoso
- 🟠 **Laranja**: Caso incerto (não analisado pelo LLM)
- **[LLM]**: Label indica que o grão foi analisado pelo LLM

## 📈 Métricas Exibidas

1. **Total Grains**: Número total de grãos detectados
2. **Good Grains**: Grãos adequados para comercialização
3. **Defective Grains**: Grãos com defeitos
4. **LLM Analyzed**: Quantos grãos foram analisados pelo LLM
5. **Uncertain Cases**: Casos próximos aos thresholds

## 🔍 Detecção de Casos Incertos

Um grão é considerado **incerto** quando:
- Circularidade entre 0.65-0.75 (threshold: 0.70)
- Aspect ratio entre 0.70-0.80 ou 1.20-1.30
- Brilho (V) entre 55-65 (threshold: 60)
- Matiz (H) entre 30-40 ou 80-90 (detecção de verde)
- Múltiplos defeitos fracos
- "Good" mas próximo dos limites

## 🧠 Modelo LLM

- **Modelo**: `meta-llama/llama-4-maverick-17b-128e-instruct`
- **Provider**: Groq
- **Capacidades**: Visão + OCR
- **Timeout**: 30 segundos por grão
- **Temperatura**: 0.3 (análise consistente)

## 🔧 Configuração da API

### Variável de Ambiente
```bash
export GROQ_API_KEY="sua_chave_aqui"
```

### No Código
A chave está atualmente hardcoded em `app.py` (linha 30). Para produção, use variáveis de ambiente.

## 📁 Saídas Geradas

### CLI (`main.py`)
- `result.jpg`: Imagem com contornos e labels
- `analysis_results.csv`: Resultados detalhados em CSV

### Web App (`app.py`)
- Visualização interativa
- Download CSV disponível
- Relatório do agrônomo AI
- Análises individuais expandíveis

## ⚠️ Tratamento de Erros

- **Timeout LLM**: Fallback para análise por regras
- **Erro na API**: Captura exceção e continua processamento
- **JSON inválido**: Parser robusto com múltiplos formatos
- **Grão sem features**: Retorna zeros e pula análise

## 🎯 Casos de Uso

### Cenários Adversos Detectados pelo LLM:
1. ✅ Sujeira/detritos na imagem
2. ✅ Texto/etiquetas
3. ✅ Manchas sutis
4. ✅ Rachaduras finas
5. ✅ Deformações complexas
6. ✅ Insetos/fragmentos
7. ✅ Cores não usuais
8. ✅ Texturas anormais

## 📊 Performance

| Modo | Tempo/grão | Precisão | Custo |
|------|-----------|----------|-------|
| All grains | 2-5s | ⭐⭐⭐⭐⭐ | Alto |
| Uncertain only | 0.5-1s | ⭐⭐⭐⭐ | Médio |
| None (rules) | <0.1s | ⭐⭐⭐ | Zero |

## 🛠️ Desenvolvimento

### Estrutura de Arquivos
```
src/demeter_ml/
├── processing.py    # Pipeline integrada
├── analysis.py      # Regras + detecção de incerteza
├── features.py      # Extração de features
├── llm.py          # Integração com Groq
├── main.py         # CLI
└── app.py          # Interface Streamlit
```

### Adicionar Novos Critérios de Incerteza

Edite `analysis.py`, função `is_uncertain_case()`:

```python
# Adicione nova condição
if nova_feature_proxima_threshold:
    uncertain = True
```

### Modificar Prompt do LLM

Edite `llm.py`, função `analyze_single_grain_with_llm()`, variável `text_prompt`.

## 📝 Logs

A pipeline imprime logs de progresso:
```
Step 1/4: Preprocessing image...
Step 2/4: Segmenting grains with Watershed...
Step 3/4: Extracting features and analyzing 15 grains...
  Grain 1/15: Using LLM analysis...
  Grain 2/15: Using LLM analysis...
Step 4/4: Analysis complete!
  LLM analysis used for 15/15 grains.
```

## 🔐 Segurança

- ⚠️ **Não commite** chaves API no código
- Use variáveis de ambiente em produção
- Considere rate limiting da API Groq

## 📞 Suporte

Para problemas:
1. Verifique se a chave API está configurada
2. Teste com `--llm-mode none` para isolar problemas de rede
3. Verifique logs para erros específicos
