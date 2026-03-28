"""
BioBERT - Medical NER via HuggingFace Inference API.
No local model downloads — all inference runs through API calls.
"""
import re
import os
import logging
import requests
import json
from django.conf import settings

logger = logging.getLogger(__name__)

# HuggingFace Inference API URLs (try router first, then legacy)
HF_API_URLS = [
    "https://router.huggingface.co/hf-inference/models/dmis-lab/biobert-base-cased-v1.2",
]

# Fallback regex patterns removed per requirement: "not trying to have anything rule based"

# Entity label mapping from HuggingFace NER models to our categories
BIOBERT_LABEL_MAP = {
    'B-Disease': 'disease', 'I-Disease': 'disease',
    'B-Chemical': 'medication', 'I-Chemical': 'medication',
    'B-Gene': 'disease', 'I-Gene': 'disease',
    'LABEL_1': 'disease', 'LABEL_2': 'medication',
    'LABEL_0': 'disease',  # For alvaroalon2/biobert_diseases_ner
}


class BertEngine:
    """
    Medical NER engine using HuggingFace Inference API for BioBERT.
    All inference runs remotely — no local model downloads.
    """

    # Models to query via HuggingFace Inference API
    # Specialized biomedical NER model (actually fine-tuned for NER)
    NER_MODEL = "d4data/biomedical-ner-all"

    def __init__(self):
        ai_config = getattr(settings, 'AI_CONFIG', {})
        self.hf_token = ai_config.get('HF_API_KEY', '') or os.getenv('HF_API_KEY', '')
        self.api_available = bool(self.hf_token)
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}

        if self.api_available:
            logger.info("HuggingFace Inference API configured for BioBERT NER")
        else:
            logger.warning("No HF_API_KEY found. Using rule-based NER fallback.")

    def extract_entities(self, text):
        """Extract medical entities — STRICTLY API ONLY. No rule-based logic."""
        if self.api_available:
            api_entities = self._hf_api_ner(text)
            if api_entities:
                return api_entities
        
        # If API is completely unavailable or crashes, return empty sets (strict no-rules policy)
        return {'disease': [], 'symptom': [], 'medication': [], 'procedure': [], 'lab_test': []}

    def _hf_api_ner(self, text):
        """Call HuggingFace Inference API for biomedical NER."""
        chunks = self._chunk_text(text, max_chars=1500)
        all_entities = {'disease': set(), 'symptom': set(), 'medication': set(),
                        'procedure': set(), 'lab_test': set()}

        for chunk in chunks:
            for base_url in HF_API_URLS:
                # If user passed a full model URL, use it directly, else append NER_MODEL
                if "models/" in base_url:
                    url = base_url
                else:
                    url = f"{base_url}/{self.NER_MODEL}"
                    
                try:
                    response = requests.post(
                        url, headers=self.headers,
                        json={"inputs": chunk, "options": {"wait_for_model": True}},
                        timeout=300
                    )

                    if response.status_code == 200:
                        results = response.json()
                        if isinstance(results, list):
                            self._process_ner_results(results, all_entities)
                        break  # Success, skip other URLs
                    elif response.status_code in (403, 410):
                        logger.info(f"HF URL {base_url} returned {response.status_code}, trying next...")
                        continue  # Try next URL
                    elif response.status_code == 503:
                        logger.info("Model loading on HuggingFace...")
                        break
                    else:
                        logger.warning(f"HF API returned {response.status_code}: {response.text[:200]}")
                        # Fallback to a known working BioBERT Token Classification model if base model fails (400 Bad Request)
                        if response.status_code == 400 and "biobert-base" in url:
                            logger.info("Base BioBERT model failed token classification. Attempting known BioBERT NER fine-tune...")
                            fallback_url = "https://router.huggingface.co/hf-inference/models/alvaroalon2/biobert_diseases_ner"
                            fb_resp = requests.post(fallback_url, headers=self.headers, json={"inputs": chunk}, timeout=300)
                            if fb_resp.status_code == 200:
                                self._process_ner_results(fb_resp.json(), all_entities)
                                break
                        continue
                except requests.exceptions.Timeout:
                    logger.warning(f"HF API timeout on {base_url}")
                    continue

        return {k: sorted(list(v)) for k, v in all_entities.items() if v}

    def _process_ner_results(self, results, all_entities):
        """Process NER results from HuggingFace API."""
        # Handle nested list format
        if results and isinstance(results[0], list):
            results = results[0]

        current_entity = ""
        current_label = ""

        for item in results:
            if not isinstance(item, dict):
                continue

            word = item.get('word', '').replace('##', '').strip()
            label = item.get('entity_group', item.get('entity', ''))
            score = item.get('score', 0)

            if score < 0.5:
                continue

            # Map biomedical NER labels to our categories
            category = self._map_label_to_category(label)
            if not category:
                continue

            # Handle B-I-O tagging (beginning vs inside)
            if label.startswith('B-') or (label != current_label):
                if current_entity and current_label:
                    mapped = self._map_label_to_category(current_label)
                    if mapped and mapped in all_entities:
                        all_entities[mapped].add(current_entity.strip())
                current_entity = word
                current_label = label
            else:
                current_entity += " " + word if not word.startswith('##') else word

        # Don't forget the last entity
        if current_entity and current_label:
            mapped = self._map_label_to_category(current_label)
            if mapped and mapped in all_entities:
                all_entities[mapped].add(current_entity.strip())

    def _map_label_to_category(self, label):
        """Map NER model labels to our entity categories."""
        label_lower = label.lower().replace('b-', '').replace('i-', '')
        mapping = {
            'disease': 'disease', 'condition': 'disease', 'disorder': 'disease',
            'sign_symptom': 'symptom', 'symptom': 'symptom',
            'medication': 'medication', 'drug': 'medication', 'chemical': 'medication',
            'therapeutic_procedure': 'procedure', 'procedure': 'procedure',
            'diagnostic_procedure': 'procedure',
            'lab_value': 'lab_test', 'lab': 'lab_test', 'test': 'lab_test',
            'biological_structure': 'disease',  # anatomy often relates to disease
            'age': None, 'sex': None, 'duration': None, 'date': None,
            'clinical_event': 'procedure', 'outcome': None,
        }
        return mapping.get(label_lower, BIOBERT_LABEL_MAP.get(label))

    def _chunk_text(self, text, max_chars=1500):
        """Split text into chunks for API calls."""
        if len(text) <= max_chars:
            return [text]
        sentences = text.replace('\n', '. ').split('. ')
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) > max_chars:
                if current:
                    chunks.append(current)
                current = s
            else:
                current = current + ". " + s if current else s
        if current:
            chunks.append(current)
        return chunks or [text[:max_chars]]


# Singleton
_bert_engine = None

def get_bert_engine():
    global _bert_engine
    if _bert_engine is None:
        _bert_engine = BertEngine()
    return _bert_engine
