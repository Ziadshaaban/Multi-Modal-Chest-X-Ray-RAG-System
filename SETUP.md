# Setup & Execution Guide

## Quick Setup (Local Machine)

### 1. Clone & Install Dependencies
```bash
cd DSAI-413-Assignment2
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys:
#  - GOOGLE_API_KEY (for Gemini API — get from https://aistudio.google.com/)
#  - KAGGLE_USERNAME, KAGGLE_KEY (from ~/.kaggle/kaggle.json)
```

### 3. Download Dataset
```bash
python src/data/download.py
```

This downloads the MIMIC-CXR dataset from Kaggle (~2-10 GB). Requires Kaggle credentials.

### 4. Preprocess Data
```bash
python src/data/preprocess.py
```

Creates `data/processed/corpus.csv` (train/index set) and `data/processed/eval.csv` (test set).

### 5. Generate QA Dataset
```bash
python src/data/qa_builder.py
```

Uses Gemini API to generate Q&A pairs from reports. Creates `data/qa/qa_dataset.csv`.

### 6. Build Retrieval Indexes
```bash
# CLIP index (runs locally on your GPU/CPU)
python src/models/clip_index.py --mode build --corpus_path data/processed/corpus.csv
```

## Cloud GPU Setup (Kaggle/Colab)

For heavy models (MedGemma, ColPali):

### On Kaggle/Colab
1. Upload the notebook: `notebooks/colab_heavy_models.ipynb`
2. Execute all cells:
   - Loads MedGemma 4B on the free T4 GPU
   - Generates reports for eval set
   - Builds ColPali index (if available)
   - Saves artifacts to `results/`

3. Download the generated CSVs and artifacts back to your local repo

## Running the Demo

### Streamlit App (Requires GPU or API)
```bash
streamlit run app/streamlit_app.py
```

Opens a web interface with two modes:
1. **Report Generation:** Upload image → select approach → see generated report
2. **QA Mode:** Upload image → ask question → see retrieved context + answer

**On GTX 1650 (local):**
- Report generation & QA both use local CLIP retrieval (fast)
- MedGemma falls back to API/cached-artifact path (can also run on Colab)

**On Colab/Kaggle T4 (GPU):**
- Full system works with MedGemma 4B + ColPali
- Use `cloudflared` or `localtunnel` to expose the Streamlit port for recording demo video

```bash
# From within Colab:
!pip install streamlit-cloudflared
streamlit run app/streamlit_app.py
```

## Testing Individual Components

### Test MedGemma (requires GPU)
```bash
python src/models/medgemma.py --mode generate --image path/to/xray.jpg
python src/models/medgemma.py --mode qa --image path/to/xray.jpg --question "Is there pneumonia?"
```

### Test CLIP Retrieval
```bash
# Build index
python src/models/clip_index.py --mode build --corpus_path data/processed/corpus.csv

# Test retrieval
python src/models/clip_index.py --mode retrieve --query_image path/to/xray.jpg --k 5
```

### Test Report Generation Pipeline
```bash
python src/pipelines/report_generation.py --image path/to/xray.jpg --approach direct
python src/pipelines/report_generation.py --image path/to/xray.jpg --approach retrieval --retriever clip
```

### Test QA Pipeline
```bash
python src/pipelines/qa_rag.py --image path/to/xray.jpg --question "Is there pneumonia?"
python src/pipelines/qa_rag.py --image path/to/xray.jpg --question "Is there pneumonia?" --ablation
```

## Troubleshooting

### "CUDA out of memory"
- Use 4-bit quantization: set `MEDGEMMA_USE_4BIT=true` in `.env`
- Reduce `MAX_REPORT_LENGTH` in `config.py`
- Use Colab/Kaggle free GPU instead

### "Image not found" when building CLIP index
- Make sure `corpus.csv` has a valid `image_path` column pointing to actual image files
- Or create dummy paths and handle gracefully in the code

### "Gemini API error" when generating QA
- Check `GOOGLE_API_KEY` in `.env`
- Make sure API is enabled in Google Cloud console
- Check rate limits (free tier has limits)

### MedGemma not loading
- On GTX 1650 (4GB VRAM): skip local loading, use Colab/Kaggle or API backend
- Make sure `HF_TOKEN` is set if the model is gated
- Check Hugging Face model access permissions

## File Structure After Setup

```
DSAI-413-Assignment2/
├── data/
│   ├── raw/               # Kaggle dataset (after download)
│   ├── processed/
│   │   ├── corpus.csv     # Train/index set
│   │   └── eval.csv       # Test set
│   └── qa/
│       ├── qa_dataset.csv # Generated QA pairs
│       └── README.md      # Generation documentation
├── results/
│   ├── clip_index/        # CLIP embeddings & index
│   │   ├── clip.index     # FAISS index
│   │   └── metadata.pkl
│   ├── generated_reports.csv # Reports from MedGemma (from cloud)
│   └── comparison_tables/     # Evaluation results
└── ...
```

## Next Steps

1. Download data & preprocess
2. Generate QA dataset
3. Build CLIP index locally
4. (Cloud) Run MedGemma generation & ColPali indexing on Kaggle/Colab
5. Launch Streamlit demo
6. Record demo video
7. Write final report with comparison tables
