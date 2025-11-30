# Demeter ML: Análise Inteligente de Grãos 🌱

**Demeter ML** é uma solução avançada para análise e classificação de qualidade de grãos (especificamente soja) utilizando uma abordagem híbrida que combina **Visão Computacional (OpenCV)** e **Large Language Models (LLMs)**.

O sistema é capaz de detectar defeitos, classificar grãos e gerar relatórios técnicos detalhados, oferecendo alta precisão ao unir a rapidez de algoritmos baseados em regras com a capacidade cognitiva de modelos de IA generativa.

---

## 🚀 Funcionalidades Principais

*   **Análise Híbrida Inteligente**: Combina algoritmos clássicos de processamento de imagem (regras de forma, cor, textura) com a análise visual de LLMs (via Groq API) para resolver casos complexos e reduzir falsos positivos.
*   **Múltiplas Interfaces**:
    *   **Web App (Streamlit)**: Interface visual interativa para upload e análise em tempo real.
    *   **CLI (Linha de Comando)**: Ferramenta para processamento em lote e automação.
    *   **API REST (Serverless)**: Endpoint escalável hospedado na AWS para integração com outros sistemas.
*   **Classificação Detalhada**: Identifica grãos quebrados, fermentados, ardidos, imaturos e outras avarias.
*   **Infraestrutura como Código**: Deploy completo na AWS (Lambda, S3, API Gateway) gerenciado via Terraform.

---

## 🧠 Como Funciona a Lógica Híbrida

O diferencial do Demeter ML é seu sistema de decisão em camadas (detalhado em `HYBRID_LOGIC.md`):

1.  **Visão Computacional (OpenCV)**: Realiza a segmentação dos grãos, extrai características métricas (circularidade, área, cor média) e aplica regras rígidas de classificação. É rápido e eficiente para casos óbvios.
2.  **LLM (Groq/Llama Vision)**: Atua como um "especialista humano". Analisa visualmente os grãos classificados como incertos ou defeituosos pelas regras, fornecendo uma segunda opinião baseada em contexto visual.
3.  **Motor de Decisão**: Um algoritmo pondera as duas análises. Se o LLM tiver alta confiança em discordar das regras (ex: identificar que uma "mancha" é apenas sombra), o sistema ajusta a classificação final.

---

## 📦 Instalação e Configuração Local

### Pré-requisitos
*   Python 3.10+
*   Conta na Groq (para chave de API do LLM)
*   AWS CLI configurado (opcional, para deploy)
*   Terraform (opcional, para deploy)

### Passos

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/vinifranco48/demeter_ml.git
    cd demeter_ml
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    # Ou se usar uv/poetry
    uv sync
    ```

4.  **Configure as variáveis de ambiente:**
    Crie um arquivo `.env` na raiz:
    ```env
    GROQ_API_KEY=sua_chave_aqui
    ```

---

## 💻 Como Usar

### 1. Interface Web (Streamlit)
A maneira mais fácil de testar.
```bash
streamlit run src/demeter_ml/app.py
```
Acesse `http://localhost:8501` no navegador.

### 2. Linha de Comando (CLI)
Para analisar uma imagem específica:
```bash
python -m demeter_ml.main caminho/para/imagem.jpg --output resultado.json
```
Opções:
*   `--llm-mode`: `all` (analisa todos com LLM), `uncertain` (apenas duvidosos), `none` (apenas regras).
*   `--save-visual`: Salva a imagem com as anotações dos grãos.

### 3. API (Se deployada)
Envie um POST com a imagem binária:
```bash
curl -X POST https://seu-api-id.execute-api.us-east-2.amazonaws.com/upload \
  -H "Content-Type: image/jpeg" \
  --data-binary "@imagem.jpg"
```

---

## ☁️ Arquitetura Cloud (AWS)

O projeto utiliza uma arquitetura **Serverless** para escalar automaticamente e reduzir custos.

*   **API Gateway**: Ponto de entrada REST.
*   **AWS Lambda**:
    *   `Sync Processor`: Executa o código Python (OpenCV + Lógica Híbrida) e retorna o resultado.
*   **Amazon S3**:
    *   `raw-images`: Armazena imagens originais.
    *   `processed-data`: Armazena relatórios JSON e imagens processadas.
*   **Terraform**: Todo o provisionamento é automatizado na pasta `/terraform`.

### Deploy
```bash
cd terraform
terraform init
terraform apply
```

---

## 📂 Estrutura do Projeto

```
demeter_ml/
├── src/demeter_ml/         # Código fonte principal
│   ├── app.py              # Interface Streamlit
│   ├── main.py             # CLI Entrypoint
│   ├── processing.py       # Pipeline de processamento e orquestração
│   ├── grain_classifier.py # Lógica de Visão Computacional (Regras)
│   ├── llm.py              # Integração com Groq API
│   └── ...
├── terraform/              # Infraestrutura como Código (AWS)
├── tests/                  # Testes unitários
├── API_DOCUMENTATION.md    # Documentação técnica da API
├── HYBRID_LOGIC.md         # Explicação detalhada da IA Híbrida
└── README.md               # Este arquivo
```

---

## 🛡️ Status do Projeto
Atualmente em fase de **Desenvolvimento/MVP**.
*   ✅ Detecção de grãos (Segmentação)
*   ✅ Classificação por Regras
*   ✅ Integração com LLM (Groq)
*   ✅ Interface Web
*   ✅ Deploy AWS Básico
*   🚧 Autenticação na API (Próximos passos)
*   🚧 Otimização de latência do Lambda

---
**Desenvolvido por Vinicius Franco**
