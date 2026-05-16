# Multi-Modal Chest X-Ray Intelligence System
## Dual-Mode: Report Generation & QA

A comprehensive system for chest X-ray analysis featuring:
- **Report Generation Mode**: Generate structured radiology reports from chest X-ray images
- **QA Mode**: Retrieval-augmented question answering over X-ray images and reports

### Key Models
- **MedGemma 4B**: Generative vision-language model for medical image understanding
- **ColPali**: Multi-vector document/image retriever (with optional fine-tuning)
- **CLIP**: Lightweight image-text retriever (baseline comparison)

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/DSAI-413-Assignment2.git
cd DSAI-413-Assignment2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

Required for QA dataset generation:
- `GOOGLE_API_KEY`: For Gemini API (QA pair generation) — get from [Google AI Studio](https://aistudio.google.com/)
- `HF_TOKEN`: For Hugging Face model access (optional, for gated models)
- `KAGGLE_USERNAME` & `KAGGLE_KEY`: For dataset download from Kaggle

Get Kaggle credentials:
1. Go to [kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. Click "Create New API Token"
3. Save the `kaggle.json` file or export the credentials to `.env`

### 4. Download dataset
```bash
python src/data/download.py
```

This downloads the MIMIC-CXR dataset from Kaggle (~2-10 GB depending on subset).

---

## Quick Start

### Option A: Local Setup (for exploration & light testing)
Works best with CLIP-based retrieval (runs on 4GB VRAM).

```bash
# 1. Prepare data (creates train/eval splits)
python src/data/preprocess.py

# 2. Generate QA dataset from reports
python src/data/qa_builder.py

# 3. Launch Streamlit demo (local CLIP retrieval)
streamlit run app/streamlit_app.py
```

### Option B: Cloud GPU Setup (Kaggle/Colab, recommended for full system)
For MedGemma generation and ColPali fine-tuning, use the provided notebook:

```bash
# Run on Kaggle or Google Colab:
# 1. Upload notebooks/colab_heavy_models.ipynb
# 2. Execute all cells (handles MedGemma generation, ColPali indexing)
# 3. Output CSVs are saved to results/
```

Then combine local + cloud artifacts:
```bash
# Local: run CLIP indexing
python src/models/clip_index.py

# Local: run Streamlit with artifacts from cloud
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
├── config.py                       # Configuration & paths
├── .env.example                    # Environment template
├── requirements.txt
│
├── data/
│   ├── raw/                        # Kaggle dataset (gitignored)
│   ├── processed/                  # Cleaned & split CSVs
│   └── qa/                         # Generated QA dataset + logs
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   └── colab_heavy_models.ipynb   # Cloud GPU pipeline
│
├── src/
│   ├── data/
│   │   ├── download.py            # Kaggle download
│   │   ├── preprocess.py          # Clean, parse, split
│   │   └── qa_builder.py          # QA generation via LLM API
│   ├── models/
│   │   ├── medgemma.py            # MedGemma inference
│   │   ├── colpali_index.py       # ColPali retriever
│   │   ├── clip_index.py          # CLIP retriever
│   │   └── colpali_finetune.py   # ColPali LoRA fine-tune (stretch)
│   ├── pipelines/
│   │   ├── report_generation.py   # Report gen: MedGemma vs retrieval
│   │   └── qa_rag.py              # QA: retrieval + MedGemma
│   ├── eval/
│   │   ├── metrics.py             # NLG & retrieval metrics
│   │   └── run_eval.py            # Batch evaluation
│   └── utils.py                   # Helper functions
│
├── app/
│   └── streamlit_app.py           # Dual-mode web UI
│
├── results/
│   ├── generated_reports.csv      # MedGemma outputs (from cloud)
│   ├── comparison_tables/         # Evaluation results
│   └── figures/
│
└── report/
    └── report.md                  # Final report
```

---

## Dual-Mode System

### Mode 1: Report Generation

**Problem:** Given a chest X-ray image, generate a structured medical report.

**Approaches compared:**
1. **MedGemma-Direct**: Image → VLM → report (generative)
2. **Retrieval-Based**: Image → retrieve similar reports → return/synthesize (retrieval)

**How to use:**
```python
from src.pipelines.report_generation import generate_report_direct, generate_report_retrieval

# Approach A: Direct generation
report_a = generate_report_direct(image_path)

# Approach B: Retrieval-based
report_b = generate_report_retrieval(image_path, retriever="clip")  # or "colpali"
```

**In Streamlit:** Upload an X-ray image → see both approaches side-by-side.

### Mode 2: QA (RAG)

**Problem:** Given an X-ray image and a clinical question, answer with retrieved context.

**Retrievers compared:**
- **CLIP**: Fast, runs locally (4GB VRAM)
- **ColPali**: Multi-vector, better semantics, requires cloud GPU

**Flow:**
1. Index corpus reports with retriever
2. Embed query image & find top-k similar reports
3. Send image + question + context to MedGemma → grounded answer
4. Ablate: with/without context to show RAG value

**How to use:**
```python
from src.pipelines.qa_rag import answer_question

answer = answer_question(
    image_path="path/to/xray.jpg",
    question="Is there pneumonia?",
    retriever="colpali",
    use_context=True
)
```

**In Streamlit:** Upload image + type question → select retriever → see answer + retrieved context.

---

## Data & QA Dataset Creation

### Dataset: MIMIC-CXR
- Source: Kaggle (`simhadrisadaram/mimic-cxr-dataset`)
- ~371K chest X-rays + free-text radiology reports
- **Working subset:** ~1000 images (800 train corpus, 150 test eval)

### QA Dataset Creation
Since no QA dataset is provided, we create one by:
1. Taking each report from the corpus
2. Prompting Gemini API with the report to generate 3–5 diverse QA pairs
3. Constraining answers to be grounded in the report text

**Question types:**
- Presence/absence of findings (e.g., "Is there atelectasis?")
- Anatomical location (e.g., "Where is the opacity located?")
- Severity assessment (e.g., "How severe is the cardiomegaly?")
- Comparisons (e.g., "Is this normal or abnormal?")

**Output:** `data/qa/qa_dataset.csv` with 3000–5000 QA pairs (3–5 per study).

**Documentation:** See `data/qa/README.md` for generation method, prompt template, and quality notes.

---

## Running Evaluations

```bash
# Generate reports (MedGemma on cloud GPU, saved to results/generated_reports.csv)
python src/pipelines/report_generation.py

# Build indexes (ColPali on cloud, CLIP locally)
python src/models/colpali_index.py
python src/models/clip_index.py

# Run QA on test set
python src/pipelines/qa_rag.py --mode eval

# Compute metrics & create comparison tables
python src/eval/run_eval.py

# View results
ls results/comparison_tables/
```

---

## Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

**Features:**
- Upload X-ray image
- Choose mode: **Report Generation** or **QA**
- In Report Gen: see MedGemma-direct vs CLIP-retrieval vs ColPali-retrieval side-by-side
- In QA: type a question, select retriever, see retrieved context + grounded answer
- Download results as JSON/CSV

---

## Stretch Goals

### ColPali Fine-Tuning
Fine-tune ColPali on chest X-ray (image, report) pairs using LoRA:
```bash
python src/models/colpali_finetune.py --num_epochs 3 --lora_r 32
```

Compare retrieval Recall@k before/after fine-tuning.

---

## Hardware Notes

| Component | Local (4GB GTX 1650) | Cloud (Kaggle/Colab T4) |
|-----------|----------------------|------------------------|
| CLIP indexing & retrieval | ✓ (fast) | — |
| MedGemma generation | ✗ (OOM) | ✓ |
| ColPali indexing | ✗ (slow/OOM) | ✓ |
| ColPali fine-tune | ✗ | ✓ (with LoRA) |
| Streamlit app | ✓ (uses artifacts) | ✓ |

**Recommendation:** Use Kaggle free GPU notebooks for heavy compute, commit artifacts, run Streamlit locally or on cloud.

---

## Results & Report

After running all evaluations:

1. **Comparison Tables** (`results/comparison_tables/`)
   - Report generation: BLEU, ROUGE, BERTScore, clinical metrics
   - Retrieval: Recall@k, MRR for ColPali vs CLIP
   - QA: token-F1, BERTScore with/without context

2. **Final Report** (`report/report.md`)
   - Architecture overview
   - Model choices & justifications
   - QA dataset creation details
   - Comparison insights & limitations

---

## Video Demo

Record a 5–15 minute video showing:
1. **Report Generation Mode:** upload image → see both approaches
2. **QA Mode:** upload image → answer multiple questions → show context
3. **Model comparison:** walk through key metrics
4. **System in action:** live Streamlit demo on cloud GPU

---

## Citation & References

- **MedGemma**: [Google DeepMind](https://deepmind.google/technologies/medgemma/)
- **ColPali**: [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)
- **MIMIC-CXR**: [MIT-LCP MIMIC-CXR](https://github.com/MIT-LCP/mimic-cxr)
- **Starter Repo**: [LightVED-prhlt/MIMIC-CXR-VQA-Dataset_Creation](https://github.com/LightVED-prhlt/MIMIC-CXR-VQA-Dataset_Creation)

---

## License

This project is for educational purposes as part of DSAI 413.
