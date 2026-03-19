"""Extractive QA model for precise span extraction from document context.

Uses a SQuAD 2.0-trained model that can:
- Extract exact answer spans from context (no hallucination)
- Detect unanswerable questions (trained on SQuAD 2.0 null examples)
- Provide calibrated confidence scores
"""

import logging
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)


class ExtractiveQA:
    """RoBERTa-based extractive QA with unanswerable detection."""

    def __init__(self, model_name: str = "deepset/roberta-base-squad2",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self):
        """Load model and tokenizer directly."""
        logger.info(f"Loading extractive QA model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.model.eval()
        logger.info("Extractive QA model loaded.")

    def answer(self, question: str, chunks: list[dict],
               no_answer_threshold: float = 0.4) -> dict:
        """Extract answer from multiple chunks, pick the best.

        Args:
            question: The user question
            chunks: List of dicts with 'text', 'chunk_id', 'page_number', 'chunk_type'
            no_answer_threshold: If best score < this, mark as unanswerable

        Returns:
            dict with answer, confidence, is_unanswerable, source_chunk_id
        """
        if self.model is None:
            self.load()

        if not chunks:
            return {
                "answer": "UNANSWERABLE: No relevant content found.",
                "confidence": 0.0,
                "is_unanswerable": True,
                "source_chunk_id": None,
            }

        # Run extractive QA on each chunk independently
        candidates = []
        for chunk in chunks:
            text = chunk["text"]
            if not text.strip():
                continue
            try:
                result = self._extract_from_context(question, text[:2000])
                result["chunk_id"] = chunk.get("chunk_id", "")
                result["page_number"] = chunk.get("page_number", 0)
                candidates.append(result)
            except Exception as e:
                logger.warning(f"Extractive QA failed on chunk {chunk.get('chunk_id')}: {e}")

        if not candidates:
            return {
                "answer": "UNANSWERABLE: Could not extract an answer.",
                "confidence": 0.0,
                "is_unanswerable": True,
                "source_chunk_id": None,
            }

        # Pick the candidate with highest score
        best = max(candidates, key=lambda c: c["score"])

        # Empty answer or low score = unanswerable
        is_unanswerable = (
            not best["answer"].strip()
            or best["score"] < no_answer_threshold
        )

        if is_unanswerable:
            return {
                "answer": "UNANSWERABLE: The document does not contain this information.",
                "confidence": round(1.0 - best["score"], 2) if best["score"] < 0.5 else 0.9,
                "is_unanswerable": True,
                "source_chunk_id": best["chunk_id"],
            }

        return {
            "answer": best["answer"].strip(),
            "confidence": round(min(best["score"], 1.0), 2),
            "is_unanswerable": False,
            "source_chunk_id": best["chunk_id"],
        }

    def _extract_from_context(self, question: str, context: str) -> dict:
        """Run extractive QA on a single context string."""
        inputs = self.tokenizer(
            question, context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0]

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # CLS token logits represent "no answer" score
        null_score = (start_logits[0] + end_logits[0]).item()

        # Find best answer span (excluding CLS)
        start_logits[0] = -1e9
        end_logits[0] = -1e9

        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()

        # Ensure valid span
        if end_idx < start_idx:
            end_idx = start_idx

        # Cap span length
        if end_idx - start_idx > 50:
            end_idx = start_idx + 50

        answer_score = (start_logits[start_idx] + end_logits[end_idx]).item()

        # Convert token indices to character offsets in context
        # offset_mapping maps each token to (start_char, end_char) in the input
        # We need offsets that map to the context portion only
        token_ids = inputs["input_ids"][0]
        sep_positions = (token_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]

        if len(sep_positions) >= 2:
            context_start_token = sep_positions[0].item() + 1  # After first SEP (for RoBERTa: </s></s>)
            # For RoBERTa: [CLS] question [SEP][SEP] context [SEP]
        else:
            context_start_token = sep_positions[0].item() + 1 if len(sep_positions) > 0 else 0

        # Extract answer text using offset mapping
        if start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
            return {"answer": "", "score": 0.0}

        char_start = offset_mapping[start_idx][0].item()
        char_end = offset_mapping[end_idx][1].item()

        # The offset mapping is relative to the full input (question + context)
        # Decode tokens directly for reliability
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        # Clean table/PDF formatting artifacts
        answer = answer.replace(" |", "").replace("| ", "").strip(" |")
        answer = " ".join(answer.split())  # Collapse whitespace/newlines

        # Score: difference between answer score and null score
        # Higher = more confident there IS an answer
        score_diff = answer_score - null_score
        # Normalize to 0-1 range using sigmoid-like mapping
        import math
        confidence = 1.0 / (1.0 + math.exp(-score_diff / 3.0))

        return {"answer": answer, "score": confidence}
