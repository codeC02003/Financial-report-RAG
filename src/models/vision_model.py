"""Vision-Language model wrapper for document understanding."""

import gc
import logging
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 768  # Resize page images to save memory


def _resize_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize image so longest side is max_size, preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


class VisionLanguageModel:
    """Wraps Qwen2-VL for multimodal document QA."""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
                 device: str = "cpu", max_new_tokens: int = 256,
                 temperature: float = 0.1):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.processor = None
        self.model_name = model_name

    def load(self):
        """Load model and processor."""
        logger.info(f"Loading {self.model_name} on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        # Always keep on CPU to avoid MPS OOM
        self.model.eval()
        logger.info("Model loaded on CPU.")

    def _generate(self, inputs: dict) -> dict:
        """Run generation and decode the answer."""
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_length:]
        answer = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Clean up repetitive output (common with small models)
        answer = self._clean_repetition(answer)

        is_unanswerable = answer.upper().startswith("UNANSWERABLE")
        confidence = self._estimate_confidence(generated_ids.shape[1], is_unanswerable)

        # Free memory
        del output_ids, generated_ids, inputs
        gc.collect()

        return {
            "answer": answer,
            "is_unanswerable": is_unanswerable,
            "confidence": confidence,
        }

    def answer_with_image(self, question: str, page_image: Image.Image,
                          context_text: str = "") -> dict:
        """Answer a question given a page image and optional text context."""
        if self.model is None:
            self.load()

        # Resize image to save memory
        page_image = _resize_image(page_image, MAX_IMAGE_SIZE)

        system_prompt = (
            "You are a financial document analyst. Answer questions about the document "
            "based on what you can see in the image and the provided text context. "
            "Extract specific numbers, values, and data directly. "
            "Give a clear, complete answer with relevant figures, years, and context. "
            "Keep it concise but informative (2-4 sentences). "
            "Only say 'UNANSWERABLE' if the information is truly not present."
        )

        user_content = [
            {"type": "image", "image": page_image},
            {"type": "text", "text": f"Text context:\n{context_text[:2000]}\n\nQuestion: {question}"},
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Keep on CPU — no .to(device)

        return self._generate(inputs)

    def answer_text_only(self, question: str, context: str) -> dict:
        """Answer a question using text context only (no image)."""
        if self.model is None:
            self.load()

        system_prompt = (
            "You are a precise financial document analyst.\n\n"
            "Rules:\n"
            "1. Find the EXACT numbers, names, and values in the context that answer the question.\n"
            "2. Read table data carefully — numbers are often formatted with $ signs, commas, and parentheses.\n"
            "3. When multiple years appear in a table, pick the value from the correct year column.\n"
            "4. For comparison questions, state BOTH values and the change.\n"
            "5. For percentage questions, compute: (part / total) × 100.\n"
            "6. If the question asks about something NOT in the context, say 'UNANSWERABLE'.\n"
            "7. Give a clear, complete answer with specific figures and context from the document.\n"
            "8. Use **bold** for key numbers and important terms.\n"
            "9. Use bullet points (- item) for lists when appropriate.\n"
            "10. Keep it concise but informative — 2-4 sentences."
        )

        # Give the model as much context as possible (cap to avoid OOM)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context[:5000]}\n\nQuestion: {question}\n\nAnswer:"},
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_input],
            padding=True,
            return_tensors="pt",
        )
        # Keep on CPU

        # Allow longer generation for detailed answers
        old_max = self.max_new_tokens
        self.max_new_tokens = 384
        result = self._generate(inputs)
        self.max_new_tokens = old_max
        return result

    def answer_conversational(self, question: str, context: str,
                              doc_summary: str = "") -> dict:
        """Answer open-ended/conversational questions with detailed responses."""
        if self.model is None:
            self.load()

        system_prompt = (
            "You are a financial document analyst. Answer questions based on the provided context.\n\n"
            "Rules:\n"
            "1. ONLY use facts, numbers, and data from the provided context.\n"
            "2. Do NOT invent or estimate numbers — only quote what appears in the context.\n"
            "3. Provide detailed, well-structured answers with specific data from the context.\n"
            "4. Quote specific text, numbers, and dates from the context.\n"
            "5. For questions about the document, look for the company name, "
            "filing type, and fiscal year on the first pages.\n"
            "6. For financial highlights, extract key revenue, income, and other figures.\n"
            "7. Use **bold** for key numbers and important terms.\n"
            "8. Use bullet points (- item) for lists. Structure your answer clearly.\n"
            "9. ALWAYS try to answer from the context. Only say information is not available "
            "as an absolute last resort when you truly cannot find anything relevant."
        )

        full_context = ""
        if doc_summary:
            full_context += f"Document Overview:\n{doc_summary}\n\n"
        full_context += f"Relevant Sections:\n{context[:5000]}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{full_context}\n\nQuestion: {question}"},
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_input],
            padding=True,
            return_tensors="pt",
        )

        # Allow longer generation for conversational answers
        old_max = self.max_new_tokens
        self.max_new_tokens = 512
        result = self._generate(inputs)
        self.max_new_tokens = old_max
        return result

    def elaborate(self, question: str, extracted_fact: str,
                  context: str, doc_overview: str = "") -> dict:
        """Take an extracted fact and elaborate it into a full answer using document context.

        This lets table extraction / extractive QA provide grounding data while
        the LLM produces a rich, document-quality response.
        """
        if self.model is None:
            self.load()

        system_prompt = (
            "You are a financial document analyst. You have been given a VERIFIED FACT "
            "extracted from a document, along with the surrounding context.\n\n"
            "Your job:\n"
            "1. Write a clear, detailed answer to the user's question.\n"
            "2. The EXTRACTED FACT below is accurate — use it as the core of your answer.\n"
            "3. Add relevant details, context, and explanation from the document context.\n"
            "4. For financial figures, mention the year, currency, and any year-over-year changes if visible.\n"
            "5. Use specific numbers, names, and dates from the context — never invent data.\n"
            "6. Keep the answer concise but informative (2-4 sentences for simple facts, more for trends/comparisons).\n"
            "7. Use **bold** for key numbers and important terms.\n"
            "8. Use bullet points (- item) for lists when appropriate.\n"
            "9. Do NOT say 'based on the extracted fact' or mention the extraction process — just answer naturally."
        )

        overview_section = f"DOCUMENT OVERVIEW:\n{doc_overview[:800]}\n\n" if doc_overview else ""
        user_content = (
            f"EXTRACTED FACT: {extracted_fact}\n\n"
            f"{overview_section}"
            f"DOCUMENT CONTEXT:\n{context[:4000]}\n\n"
            f"Question: {question}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_input],
            padding=True,
            return_tensors="pt",
        )

        old_max = self.max_new_tokens
        self.max_new_tokens = 384
        result = self._generate(inputs)
        self.max_new_tokens = old_max
        return result

    @staticmethod
    def _clean_repetition(text: str) -> str:
        """Remove degenerate repetition from model output."""
        # Detect repeated short patterns (e.g. "(1) (26) (1) (26) ...")
        import re
        # Find repeating patterns of 3-30 chars that repeat 3+ times
        match = re.search(r'(.{3,30}?)\1{2,}', text)
        if match:
            # Cut at the start of the repetition
            text = text[:match.start()].rstrip(" ,;:-")

        # Also detect numbered list items that repeat the same content
        lines = text.split('\n')
        seen_lines = set()
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped in seen_lines and len(stripped) > 10:
                continue
            seen_lines.add(stripped)
            clean_lines.append(line)
        text = '\n'.join(clean_lines)

        return text.strip()

    def _estimate_confidence(self, generated_length: int,
                             is_unanswerable: bool) -> float:
        """Estimate answer confidence based on generation length."""
        if generated_length <= 0:
            return 0.0
        if is_unanswerable:
            return 0.9
        base = min(1.0, 0.4 + (generated_length / 30.0))
        return round(base, 2)
