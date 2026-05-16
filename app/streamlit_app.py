"""
Multi-Modal Chest X-Ray Intelligence System — Streamlit Demo

Dual-mode interface:
  1. Report Generation: Upload image → see MedGemma-direct vs retrieval-based reports
  2. QA Mode: Upload image + question → see retrieved context + grounded answer
"""

import streamlit as st
from PIL import Image
import io
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.pipelines.report_generation import ReportGenerationPipeline
from src.pipelines.qa_rag import QARagPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🫁 Multi-Modal Chest X-Ray Intelligence System")
st.markdown("**Dual-Mode AI for Medical Image Analysis**")

with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio(
        "Select Mode",
        ["📋 Report Generation", "❓ Question Answering (RAG)"],
        index=0
    )

    st.divider()
    st.subheader("System Info")
    st.text(f"Device: {config.DEVICE}")
    st.text(f"VRAM: {config.CUDA_DEVICE_MEMORY:.1f} GB")
    st.text(f"MedGemma Backend: {config.MEDGEMMA_BACKEND}")
    st.text(f"Use 4-bit: {config.MEDGEMMA_USE_4BIT}")

    st.divider()
    with st.expander("Advanced Settings"):
        retriever_choice = st.selectbox(
            "Retriever (QA mode)",
            ["clip", "colpali"],
            index=0
        )
        k_retrieved = st.slider("Number of retrieved documents (k)", 1, 10, 3)
        max_tokens = st.slider("Max answer length (tokens)", 128, 1024, 256)
        temperature = st.slider("Temperature (generation)", 0.0, 2.0, 0.7)

@st.cache_resource
def load_report_pipeline():
    """Load report generation pipeline (cached)."""
    try:
        clip_index_path = config.RESULTS_DIR / "clip_index"
        return ReportGenerationPipeline(
            device=config.DEVICE,
            use_4bit=config.MEDGEMMA_USE_4BIT,
            medgemma_backend=config.MEDGEMMA_BACKEND,
            clip_index_path=str(clip_index_path) if clip_index_path.exists() else None,
            colpali_index_path=None,  # ColPali needs GPU >=6GB, skip locally
        )
    except Exception as e:
        st.error(f"Failed to load report pipeline: {e}")
        return None

@st.cache_resource
def load_qa_pipeline():
    """Load QA RAG pipeline (cached)."""
    try:
        clip_index_path = config.RESULTS_DIR / "clip_index"
        return QARagPipeline(
            device=config.DEVICE,
            use_4bit=config.MEDGEMMA_USE_4BIT,
            clip_index_path=str(clip_index_path) if clip_index_path.exists() else None,
            colpali_index_path=None,  # ColPali needs GPU >=6GB, skip locally
            medgemma_backend=config.MEDGEMMA_BACKEND,
        )
    except Exception as e:
        st.error(f"Failed to load QA pipeline: {e}")
        return None

if mode == "📋 Report Generation":
    st.header("📋 Report Generation Mode")
    st.markdown("Upload a chest X-ray image to generate a structured radiology report.")
    st.markdown("*Compares two approaches: generative (MedGemma) vs retrieval-based*")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        if uploaded_file:
            st.subheader("Report Options")
            report_approach = st.radio(
                "Generation Approach",
                ["Direct (MedGemma)", "Retrieval-Based (CLIP)"],
                index=0
            )

            if st.button("🔄 Generate Report", key="gen_report"):
                with st.spinner("Generating report... (this may take a moment)"):
                    try:
                        pipeline = load_report_pipeline()
                        if pipeline is None:
                            st.error("Report pipeline not initialized")
                        elif report_approach == "Direct (MedGemma)":
                            report = pipeline.generate_direct(image, max_tokens=config.MAX_REPORT_LENGTH)
                            st.markdown("### Generated Report (MedGemma-Direct)")
                            st.markdown(report)
                        else:
                            report = pipeline.generate_retrieval_based(image, retriever="clip", k=3)
                            st.markdown("### Generated Report (CLIP-Retrieval)")
                            st.markdown(report)

                    except Exception as e:
                        st.error(f"Error generating report: {e}")
                        st.info("💡 Tip: Make sure you're on a GPU-enabled environment for MedGemma.")

            st.divider()
            st.info("💡 **How it works:**\n- **Direct:** Image → VLM → Report\n- **Retrieval:** Image → Find similar → Return their reports")

elif mode == "❓ Question Answering (RAG)":
    st.header("❓ Question Answering Mode (RAG)")
    st.markdown("Ask clinical questions about a chest X-ray. The system retrieves relevant context and provides grounded answers.")
    st.markdown("*Uses retrieval-augmented generation (RAG) with MedGemma*")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="qa_uploader"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        if uploaded_file:
            st.subheader("Ask a Question")
            question = st.text_area(
                "Enter your clinical question",
                placeholder="e.g., Is there pneumonia present? Where is the opacity located?",
                label_visibility="collapsed",
                height=100
            )

            col_ret, col_ctx = st.columns(2)
            with col_ret:
                use_context = st.checkbox("Use retrieved context (RAG)", value=True)
            with col_ctx:
                retriever = st.selectbox("Retriever", ["clip", "colpali"], label_visibility="collapsed")

            if st.button("🔍 Answer Question", key="qa_button"):
                if not question.strip():
                    st.warning("Please enter a question")
                else:
                    with st.spinner("Generating answer... (this may take a moment)"):
                        try:
                            pipeline = load_qa_pipeline()
                            if pipeline is None:
                                st.error("QA pipeline not initialized")
                            else:
                                result = pipeline.answer_question(
                                    image,
                                    question,
                                    retriever=retriever,
                                    use_context=use_context,
                                    k=k_retrieved,
                                    max_tokens=max_tokens
                                )

                                st.success("✓ Answer generated!")

                                st.markdown("### Answer")
                                st.markdown(f"**{result['answer']}**")

                                if use_context:
                                    st.divider()
                                    st.markdown("### Retrieved Context")
                                    for i, report in enumerate(result['retrieved_reports'], 1):
                                        with st.expander(f"Context Document {i}"):
                                            st.text(report[:500] + "..." if len(report) > 500 else report)

                        except Exception as e:
                            st.error(f"Error answering question: {e}")
                            st.info("💡 Tip: Make sure you're on a GPU-enabled environment for MedGemma.")

            st.divider()
            st.info("💡 **RAG Workflow:**\n1. Embed your image\n2. Retrieve similar reports\n3. Use context to ground the answer\n4. Generate with MedGemma")

st.divider()
st.markdown("""
---
### About This System

This is a multi-modal chest X-ray analysis system built with:
- **MedGemma 4B**: Medical vision-language model
- **CLIP** & **ColPali**: Multi-modal retrievers
- **RAG**: Retrieval-augmented generation for grounded answers

**Modes:**
1. **Report Generation**: Automatically generate clinical reports
2. **QA with RAG**: Answer questions with retrieved context

**Note:** This system is for demonstration purposes. All outputs should be independently verified by qualified medical professionals.

---
*DSAI 413 - Assignment 2 | Multi-Modal Chest X-Ray Intelligence System*
""")
