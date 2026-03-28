"""
LLM Engine - Multi-model ensemble via APIs for medical diagnosis.
Models: DeepSeek R1 (API), Med42 (HuggingFace API), PMC-LLaMA (HF API)
No local model downloads — all inference runs through remote APIs.
"""
import logging
import json
import time
import os
import requests
from django.conf import settings
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class LLMEngine:
    """Multi-model LLM ensemble using remote APIs."""

    # Requested by assignment
    MODELS = ['Med42']

    # HuggingFace model IDs
    HF_MODELS = {
        'Med42': 'm42-health/Llama3-Med42-8B:featherless-ai',
    }

    def __init__(self):
        ai_config = getattr(settings, 'AI_CONFIG', {})
        self.hf_token = ai_config.get('HF_API_KEY', '') or os.getenv('HF_API_KEY', '')
        self.use_simulated = ai_config.get('USE_SIMULATED_LLM', True)

        self.has_hf = bool(self.hf_token)
        self.hf_client = InferenceClient(token=self.hf_token) if self.has_hf else None

        logger.info(f"LLM Engine: HuggingFace={'configured' if self.has_hf else 'not set'}, "
                     f"Simulated={self.use_simulated}")

    def generate_diagnosis(self, discharge_summary, entities, graph_context):
        """Generate diagnosis using available API models."""
        start_time = time.time()
        prompt = self._build_medical_prompt(discharge_summary, entities, graph_context)

        models_used = []
        responses = {}

        if self.has_hf and not self.use_simulated:
            for model_name, model_id in self.HF_MODELS.items():
                try:
                    logger.info(f"Attempting inference with {model_name} ({model_id})...")
                    result = self._call_hf_model(model_name, model_id, prompt)
                    if result:
                        responses[model_name] = result
                        models_used.append(model_name)
                        logger.info(f"{model_name} response received successfully")
                        break  # Stop at first successful model
                except Exception as e:
                    logger.warning(f"{model_name} native API failed (likely too large for free tier): {e}")

        # --- Result ---
        if responses:
            result = self._ensemble_responses(responses, entities, graph_context)
            result['models_used'] = models_used
        else:
            # Enforce strict LLM usage; no rule-based fallbacks allowed
            raise RuntimeError("LLM API (Med42) failed to generate a diagnosis. Rule-based fallbacks are strictly disabled.")

        result['processing_time'] = time.time() - start_time
        return result

    def _build_medical_prompt(self, summary, entities, context):
        """Build structured medical prompt."""
        related_diseases = context.get('related_diseases', [])
        top_diseases = ', '.join([d['disease'] for d in related_diseases[:5]]) if related_diseases else 'None identified'

        return f"""You are an expert clinical AI system for medical diagnosis. Analyze the following patient discharge summary and provide a precise diagnosis with treatment plan.

PATIENT DISCHARGE SUMMARY:
{summary[:3000]}

EXTRACTED MEDICAL ENTITIES (from BioBERT NER):
- Diseases mentioned: {', '.join(entities.get('disease', [])[:10])}
- Symptoms: {', '.join(entities.get('symptom', [])[:15])}
- Medications: {', '.join(entities.get('medication', [])[:10])}
- Procedures: {', '.join(entities.get('procedure', [])[:10])}
- Lab tests: {', '.join(entities.get('lab_test', [])[:10])}

KNOWLEDGE GRAPH CONTEXT (GraphRAG):
- Related diseases by symptom matching: {top_diseases}
- Drug interactions found: {len(context.get('drug_interactions', []))}

Provide your response in the following exact JSON format:
{{
  "primary_diagnosis": "The single most likely primary diagnosis",
  "secondary_diagnoses": ["list", "of", "secondary", "diagnoses"],
  "confidence": 0.85,
  "reasoning": "Detailed clinical reasoning explaining why this diagnosis was reached, citing specific symptoms, lab values, and imaging findings",
  "treatment_plan": "Comprehensive treatment plan with specific interventions",
  "medications": ["medication1 with dose", "medication2 with dose"],
  "procedures": ["recommended procedures"],
  "lifestyle_modifications": ["modification1", "modification2"],
  "follow_up": ["1 week: purpose", "1 month: purpose"]
}}"""

    def _call_hf_model(self, model_name, model_id, prompt):
        """Call a model via HuggingFace Hub using direct requests to the v1 chat endpoint."""
        if not self.hf_token:
            return None

        # Give it a single quick attempt so we fall back fast if the heavy model is not loaded
        try:
            api_url = "https://router.huggingface.co/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a master clinical AI system. Always format output in valid JSON as requested, with no markdown block wrappers around the JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2048,
                "temperature": 0.3
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=300) # Give 5 mins instead of 1 min
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return self._parse_json_response(content)
            else:
                logger.warning(f"HF API {model_id} returned {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.warning(f"HF API {model_id} error: {e}")

        return None

    def _parse_json_response(self, text):
        """Parse JSON from LLM response (handles markdown code blocks)."""
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        json_match = re.search(r'\{[^{}]*"primary_diagnosis"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Return as raw text
        return {'raw_response': text}

    def _ensemble_responses(self, responses, entities, graph_context):
        """Combine responses from multiple models into a final diagnosis."""
        primary = responses.get('Med42', {})
        if 'raw_response' in primary:
            # If the model failed to output JSON and returned raw text
            return {
                'primary_diagnosis': 'Unstructured Model Output',
                'secondary_diagnoses': [],
                'diagnosis_confidence': 0.0,
                'diagnosis_reasoning': primary['raw_response'],
                'treatment_plan': 'Review raw text for details.',
                'medications_recommended': [],
                'procedures_recommended': [],
                'lifestyle_modifications': [],
                'follow_up_schedule': [],
            }

        all_secondaries = set()
        all_medications = set()
        all_procedures = set()
        all_lifestyle = set()

        for model_name, resp in responses.items():
            if isinstance(resp, dict):
                for s in resp.get('secondary_diagnoses', []):
                    all_secondaries.add(s)
                for m in resp.get('medications', []):
                    all_medications.add(m)
                for p in resp.get('procedures', []):
                    all_procedures.add(p)
                for l in resp.get('lifestyle_modifications', []):
                    all_lifestyle.add(l)

        # Build reasoning string
        reasoning_parts = []
        for model_name, resp in responses.items():
            if isinstance(resp, dict) and 'raw_response' not in resp:
                reasoning_parts.append(f"ENGINE: {model_name}")
                reasoning_parts.append(f"PRIMARY DIAGNOSIS: {resp.get('primary_diagnosis', 'N/A')} (Confidence: {resp.get('confidence', 'N/A')})")
                reasoning_parts.append(f"\nCLINICAL REASONING:\n{resp.get('reasoning', 'N/A')}\n")

        result = {
            'primary_diagnosis': primary.get('primary_diagnosis', 'Undetermined'),
            'secondary_diagnoses': list(all_secondaries)[:5],
            'diagnosis_confidence': float(primary.get('confidence', 0.85)),
            'diagnosis_reasoning': '\n'.join(reasoning_parts),
            'treatment_plan': primary.get('treatment_plan', ''),
            'medications_recommended': [{'name': m, 'rationale': 'LLM recommended'} for m in all_medications],
            'procedures_recommended': list(all_procedures),
            'lifestyle_modifications': list(all_lifestyle),
            'follow_up_schedule': self._parse_followup(primary.get('follow_up', [])),
        }

        return result

    def _parse_followup(self, followup_list):
        """Parse follow-up items into structured format."""
        schedule = []
        for item in followup_list:
            if isinstance(item, str) and ':' in item:
                parts = item.split(':', 1)
                schedule.append({'timeframe': parts[0].strip(), 'type': 'Follow-up', 'purpose': parts[1].strip()})
            elif isinstance(item, str):
                schedule.append({'timeframe': 'TBD', 'type': 'Follow-up', 'purpose': item})
        return schedule or [
            {'timeframe': '1 week', 'type': 'Primary care', 'purpose': 'Post-discharge check'},
            {'timeframe': '1 month', 'type': 'Specialist', 'purpose': 'Treatment response'},
        ]

# Singleton
_llm_engine = None

def get_llm_engine():
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine()
    return _llm_engine
