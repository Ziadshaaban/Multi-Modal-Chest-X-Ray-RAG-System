"""
MedGemma 4B-IT: medical vision-language model for report generation and QA.

Loads `google/medgemma-4b-it` with optional 4-bit (NF4) quantization. Generation
defaults to greedy/deterministic for reproducible clinical outputs.

Usage:
    python src/models/medgemma.py --mode generate --image path/to/image.jpg
    python src/models/medgemma.py --mode qa --image x.jpg --question "Is there pneumonia?"
"""

import argparse
import base64
import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HuggingFaceGenerator:
    """HuggingFace Inference API backend — free, no GPU needed."""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"):
        from huggingface_hub import InferenceClient
        self.token = api_key or config.HF_TOKEN
        if not self.token:
            raise ValueError("HF_TOKEN not set in .env")
        self.client = InferenceClient(token=self.token)
        self.model = model
        logger.info(f"HuggingFace Inference API ready (model={model})")

    def _image_to_b64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    def generate_report(self, image: Image.Image, max_tokens: int = 512, **kwargs) -> str:
        prompt = (
            "You are an expert radiologist. Analyze this chest X-ray and provide a "
            "structured report with FINDINGS and IMPRESSION sections. Be clinically "
            "precise and concise."
        )
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._image_to_b64(image)}"}},
            {"type": "text", "text": prompt},
        ]}]
        result = self.client.chat_completion(messages=messages, model=self.model, max_tokens=max_tokens)
        return result.choices[0].message.content.strip()

    def answer_question(self, image: Image.Image, question: str,
                        context: Optional[str] = None, max_tokens: int = 256, **kwargs) -> str:
        if context:
            text = (
                "You are an expert radiologist. Use the retrieved similar reports as "
                "reference context, but base your answer primarily on the X-ray image.\n\n"
                f"Reference context:\n{context}\n\n"
                f"Question: {question}\nAnswer concisely."
            )
        else:
            text = f"You are an expert radiologist. Question about this chest X-ray: {question}\nAnswer concisely."
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._image_to_b64(image)}"}},
            {"type": "text", "text": text},
        ]}]
        result = self.client.chat_completion(messages=messages, model=self.model, max_tokens=max_tokens)
        return result.choices[0].message.content.strip()


class GeminiGenerator:
    """Gemini API backend with multi-key rotation and 429 retry."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        self.genai = genai
        self.model_name = model

        # Collect all keys: GOOGLE_API_KEY, GOOGLE_API_KEY2, GOOGLE_API_KEY3, ...
        keys = []
        primary = api_key or os.getenv("GOOGLE_API_KEY", "")
        if primary:
            keys.append(primary)
        for i in range(2, 10):
            k = os.getenv(f"GOOGLE_API_KEY{i}", "")
            if k:
                keys.append(k)
        if not keys:
            raise ValueError("No GOOGLE_API_KEY found in .env")
        self._keys = keys
        self._key_idx = 0
        self._init_model()
        logger.info(f"Gemini API backend ready (model={model}, keys={len(keys)})")

    def _init_model(self):
        import google.generativeai as genai
        genai.configure(api_key=self._keys[self._key_idx])
        self.model = genai.GenerativeModel(self.model_name)

    def _rotate_key(self):
        self._key_idx = (self._key_idx + 1) % len(self._keys)
        self._init_model()
        logger.info(f"Rotated to Gemini key index {self._key_idx}")

    def _call(self, parts, max_tokens: int) -> str:
        import time
        for attempt in range(len(self._keys) * 2):
            try:
                response = self.model.generate_content(
                    parts,
                    generation_config={"max_output_tokens": max_tokens},
                )
                return response.text.strip()
            except Exception as e:
                msg = str(e)
                if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                    logger.warning(f"Gemini 429 on key {self._key_idx}, rotating...")
                    self._rotate_key()
                    time.sleep(1)
                else:
                    raise
        raise RuntimeError("All Gemini API keys exhausted / rate-limited")

    def _image_to_part(self, image: Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return self.genai.protos.Part(
            inline_data=self.genai.protos.Blob(
                mime_type="image/jpeg",
                data=buf.getvalue(),
            )
        )

    def generate_report(self, image: Image.Image, max_tokens: int = 512, **kwargs) -> str:
        prompt = (
            "You are an expert radiologist. Analyze this chest X-ray and provide a "
            "structured report with FINDINGS and IMPRESSION sections. Be clinically "
            "precise and concise."
        )
        return self._call([self._image_to_part(image), prompt], max_tokens)

    def answer_question(self, image: Image.Image, question: str,
                        context: Optional[str] = None, max_tokens: int = 256, **kwargs) -> str:
        if context:
            prompt = (
                "You are an expert radiologist. Use the retrieved similar reports as "
                "reference context, but base your answer primarily on the X-ray image.\n\n"
                f"Reference context:\n{context}\n\n"
                f"Question: {question}\nAnswer concisely."
            )
        else:
            prompt = (
                "You are an expert radiologist. "
                f"Question about this chest X-ray: {question}\nAnswer concisely."
            )
        return self._call([self._image_to_part(image), prompt], max_tokens)

DEFAULT_REPORT_SYSTEM = (
    "You are an expert radiologist. Analyze the chest X-ray and provide a "
    "structured report with FINDINGS and IMPRESSION sections. Be clinically "
    "precise and concise."
)
DEFAULT_QA_SYSTEM = (
    "You are an expert radiologist. Answer the clinical question based on the "
    "image and any provided context. Be concise and factual; do not invent "
    "findings that are not visible."
)


class MedGemmaGenerator:
    def __init__(self, model_id: str = "google/medgemma-4b-it",
                 device: str = "cuda", use_4bit: bool = False,
                 torch_dtype: Optional[torch.dtype] = None):
        self.model_id = model_id
        self.device = device
        self.use_4bit = use_4bit

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        logger.info(f"Loading MedGemma ({model_id}) | device={device} | "
                   f"4bit={use_4bit} | dtype={torch_dtype}")
        self._load_model()

    def _load_model(self):
        quantization_config = None
        if self.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("Using 4-bit NF4 quantization")
            except ImportError:
                logger.warning("bitsandbytes unavailable; loading in full precision")

        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device == "cuda" else self.device,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model.eval()
            logger.info("MedGemma loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MedGemma: {e}")
            logger.error("Check HF_TOKEN and access to google/medgemma-4b-it")
            raise

    def _generate(self, messages: list, max_new_tokens: int,
                  do_sample: bool, temperature: Optional[float],
                  top_p: Optional[float]) -> str:
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            generation = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = generation[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_report(self, image: Image.Image,
                        system_prompt: Optional[str] = None,
                        user_prompt: Optional[str] = None,
                        max_tokens: int = 512, do_sample: bool = False,
                        temperature: float = 0.7, top_p: float = 0.95) -> str:
        system_prompt = system_prompt or DEFAULT_REPORT_SYSTEM
        user_prompt = user_prompt or (
            "Analyze this chest X-ray and provide a structured clinical report "
            "with FINDINGS and IMPRESSION sections."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": image},
            ]},
        ]
        return self._generate(messages, max_tokens, do_sample, temperature, top_p)

    def answer_question(self, image: Image.Image, question: str,
                        context: Optional[str] = None, max_tokens: int = 256,
                        do_sample: bool = False, temperature: float = 0.7,
                        top_p: float = 0.95) -> str:
        if context:
            user_text = (
                "Use the retrieved similar reports as reference context, but base "
                "your answer primarily on the image.\n\n"
                f"Reference context:\n{context}\n\n"
                f"Question: {question}\nAnswer concisely."
            )
        else:
            user_text = f"Question about this chest X-ray: {question}\nAnswer concisely."

        messages = [
            {"role": "system", "content": [{"type": "text", "text": DEFAULT_QA_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": image},
            ]},
        ]
        return self._generate(messages, max_tokens, do_sample, temperature, top_p)


def main():
    parser = argparse.ArgumentParser(description="MedGemma inference")
    parser.add_argument("--mode", choices=["generate", "qa"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", help="Question for QA mode")
    parser.add_argument("--context", help="Context for QA (RAG)")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--sample", action="store_true",
                        help="Sample (do_sample=True). Default is greedy/deterministic.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MEDGEMMA INFERENCE")
    logger.info("=" * 60)

    generator = MedGemmaGenerator(
        device=args.device, use_4bit=args.use_4bit or config.MEDGEMMA_USE_4BIT,
    )
    image = Image.open(args.image).convert("RGB")

    if args.mode == "generate":
        report = generator.generate_report(
            image, max_tokens=args.max_tokens, do_sample=args.sample,
        )
        logger.info("\nGENERATED REPORT:\n" + report)

    elif args.mode == "qa":
        if not args.question:
            raise ValueError("Question required for QA mode")
        answer = generator.answer_question(
            image, args.question, context=args.context,
            max_tokens=args.max_tokens, do_sample=args.sample,
        )
        logger.info(f"\nQUESTION: {args.question}")
        logger.info(f"\nANSWER:\n{answer}")


if __name__ == "__main__":
    main()
