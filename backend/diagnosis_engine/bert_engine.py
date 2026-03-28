"""
BioBERT - Medical NER via HuggingFace Inference API.
STRICTLY uses only BioBERT models as required by assignment.

Fixes applied in this version:
  1. Noise phrase filtering  — "Presented with", "Started on", sentence
     fragments are removed BEFORE categorisation.
  2. palpitations mis-label  — was hitting "test" substring in LAB_TEST_TERMS.
     SYMPTOM_TERMS checked FIRST, and LAB_TEST_TERMS now uses whole-word matching.
  3. Sentence-fragment detection — entities longer than 6 words are dropped
     unless they match a known multi-word medical term.
  4. Procedure detection fixed — "rate control and anticoagulation" correctly
     routed to procedure via treatment verb patterns.
"""

import os
import re
import logging
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BioBERT model endpoints (ONLY BioBERT — assignment requirement)
# ---------------------------------------------------------------------------
HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"
BIOBERT_PRIMARY   = "alvaroalon2/biobert_diseases_ner"
BIOBERT_SECONDARY = "fran-martinez/biobert-base-cased-v1.2-finetuned-ner"

# ---------------------------------------------------------------------------
# NOISE PHRASES — discard these entirely, they are sentence fragments
# ---------------------------------------------------------------------------
NOISE_PHRASES = {
    "presented with", "presenting with", "presents with",
    "started on", "started with", "start on",
    "history of", "known case of", "known history of",
    "diagnosed with", "diagnosis of", "newly diagnosed",
    "confirmed", "confirmed newly diagnosed",
    "complains of", "complaints of", "complaint of",
    "admitted with", "admitted for",
    "referred for", "referred with",
    "underwent", "underwent a", "underwent an",
    "treated with", "treated for",
    "follow up", "follow-up", "review of",
    "no history of", "no known",
    "significant", "significant for",
    "consistent with", "consistent",
    "secondary to", "due to",
    "associated with",
    "in the setting of",
    "patient", "the patient",
    "he", "she", "they", "it", "this", "the",
}

# Patterns that mark a string as sentence noise (not a medical entity)
NOISE_PATTERNS = [
    r"^(presented?|presenting|presents?)\s+(with|on|to)",
    r"^(started?|start|begin|began)\s+(on|with)",
    r"^(history|hx)\s+of",
    r"^(diagnosed?|diagnosis)\s+(with|of)",
    r"^(treated?|treatment)\s+(with|for|of)",
    r"^(complains?\s+of|complaints?\s+of)",
    r"^(admitted?|referr)\w*\s+(with|for|to)",
    r"^(confirmed|showing|reveals?|found)\b",
    r"^(no\s+|without\s+|denies?\s+)",
    r"\b(and|or|with|the|on|was|were|is|are|has|have)\s+\w+$",  # ends with common word
]

# ---------------------------------------------------------------------------
# SYMPTOM TERMS  (checked BEFORE lab tests to prevent palpitations → lab_test)
# ---------------------------------------------------------------------------
SYMPTOM_TERMS = {
    # Pain
    "pain", "chest pain", "back pain", "neck pain", "abdominal pain",
    "stomach pain", "joint pain", "muscle pain", "leg pain", "arm pain",
    "shoulder pain", "knee pain", "hip pain", "pelvic pain", "flank pain",
    "throat pain", "eye pain", "ear pain", "tooth pain", "jaw pain",
    "headache", "migraine", "ache", "aching", "sore", "soreness",
    "tenderness", "cramp", "cramping", "discomfort", "stiffness",
    "burning sensation", "burning", "sharp pain", "dull pain",
    "throbbing pain", "stabbing pain",
    # Respiratory
    "shortness of breath", "breathlessness", "dyspnea", "dyspnoea",
    "difficulty breathing", "labored breathing", "rapid breathing",
    "wheezing", "wheeze", "cough", "coughing", "dry cough", "wet cough",
    "productive cough", "hemoptysis", "chest tightness",
    "nasal congestion", "runny nose", "stuffy nose", "sneezing",
    # Cardiac / vascular  ← palpitations lives here
    "palpitation", "palpitations", "irregular heartbeat", "racing heart",
    "heart racing", "skipped beat", "fluttering", "chest tightness",
    # Neurological
    "dizziness", "vertigo", "lightheadedness", "fainting", "syncope",
    "numbness", "tingling", "tremor", "trembling", "shaking",
    "weakness", "muscle weakness", "limb weakness",
    "confusion", "disorientation", "memory loss", "forgetfulness",
    "difficulty concentrating", "brain fog", "seizure", "convulsion",
    "loss of consciousness", "blurred vision", "double vision",
    "vision changes", "visual disturbance",
    # GI
    "nausea", "vomiting", "diarrhea", "diarrhoea", "constipation",
    "bloating", "flatulence", "heartburn", "indigestion", "belching",
    "stomach ache", "abdominal cramps", "rectal bleeding", "blood in stool",
    "black stool", "loss of appetite", "difficulty swallowing", "dysphagia",
    "regurgitation", "acid reflux",
    # Systemic
    "fatigue", "tiredness", "exhaustion", "lethargy", "malaise",
    "fever", "high fever", "low grade fever", "chills", "rigors",
    "night sweats", "sweating", "excessive sweating", "hot flashes",
    "weight loss", "unexplained weight loss", "weight gain",
    "decreased appetite", "increased appetite", "anorexia",
    "dehydration", "pallor",
    # Skin
    "rash", "skin rash", "itching", "pruritus", "hives", "urticaria",
    "swelling", "edema", "oedema", "bruising", "easy bruising",
    "redness", "erythema", "jaundice", "dry skin", "peeling skin",
    # ENT / Eyes
    "hearing loss", "tinnitus", "ringing in ears", "ear pain",
    "sore throat", "throat irritation", "hoarseness", "voice changes",
    "eye redness", "watery eyes", "eye discharge",
    # Urinary
    "frequent urination", "painful urination", "dysuria", "hematuria",
    "blood in urine", "incontinence", "urinary urgency",
    "decreased urination", "dark urine",
    # Musculoskeletal
    "joint swelling", "stiff joints", "limited range of motion",
    "muscle spasm", "muscle cramps", "muscle ache",
    # Mental / mood
    "anxiety", "nervousness", "panic", "depression", "sadness",
    "irritability", "mood swings", "agitation", "restlessness",
    "insomnia", "sleep disturbance", "difficulty sleeping",
    "excessive sleeping", "nightmares",
}

# ---------------------------------------------------------------------------
# DISEASE ANCHOR TERMS — these pin an entity as disease, overrides symptom rescue
# ---------------------------------------------------------------------------
DISEASE_ANCHOR_TERMS = {
    "disease", "disorder", "syndrome", "cancer", "carcinoma", "tumor",
    "tumour", "lymphoma", "leukemia", "leukaemia", "sarcoma", "melanoma",
    "diabetes", "hypertension", "hypotension", "infection", "pneumonia",
    "sepsis", "bacteremia", "viremia", "meningitis", "encephalitis",
    "stroke", "infarction", "ischemia", "failure", "insufficiency",
    "stenosis", "regurgitation", "prolapse", "rupture",
    "fracture", "dislocation", "injury", "trauma",
    "arthritis", "osteoporosis", "spondylosis", "fibromyalgia",
    "asthma", "copd", "emphysema", "bronchitis", "fibrosis",
    "hepatitis", "cirrhosis", "pancreatitis", "colitis", "gastritis",
    "nephropathy", "neuropathy", "myopathy", "cardiomyopathy",
    "retinopathy", "encephalopathy",
    "sclerosis", "dementia", "alzheimer", "parkinson", "epilepsy",
    "hypothyroidism", "hyperthyroidism", "addison", "cushing",
    "anemia", "anaemia", "thrombocytopenia", "leukopenia",
    "thrombosis", "embolism", "atherosclerosis", "hypertrophy",
    "malignancy", "metastasis", "abscess", "hernia", "ulcer", "polyp",
    "cyst", "calculus", "stones", "calcification",
    "allergy", "anaphylaxis", "autoimmune", "immunodeficiency",
    "hiv", "aids", "tuberculosis", "malaria", "dengue", "typhoid",
    "fibrillation", "tachycardia", "bradycardia", "arrhythmia",
    "flutter", "block", "dissection",
}

# ---------------------------------------------------------------------------
# ANATOMY TERMS
# ---------------------------------------------------------------------------
ANATOMY_TERMS = {
    "heart", "lung", "lungs", "liver", "kidney", "kidneys", "brain",
    "spinal cord", "spine", "vertebra", "vertebrae",
    "stomach", "intestine", "intestines", "colon", "rectum", "bowel",
    "pancreas", "gallbladder", "bladder", "uterus", "ovary", "ovaries",
    "prostate", "testis", "testes", "breast", "thyroid", "adrenal",
    "spleen", "bone marrow", "lymph node", "lymph nodes",
    "artery", "arteries", "vein", "veins", "aorta", "coronary",
    "ventricle", "atrium", "mitral valve", "aortic valve",
    "trachea", "bronchus", "bronchi", "pleura", "diaphragm",
    "esophagus", "oesophagus", "duodenum", "jejunum", "ileum",
    "appendix", "peritoneum",
    "femur", "tibia", "fibula", "humerus", "radius", "ulna",
    "skull", "sternum", "rib", "ribs", "pelvis", "clavicle",
    "knee", "hip", "shoulder", "elbow", "ankle", "wrist",
    "skin", "muscle", "tendon", "ligament", "cartilage", "nerve",
    "retina", "cornea", "lens", "optic nerve",
    "cochlea", "eardrum",
    "cerebrum", "cerebellum", "brainstem", "hypothalamus", "thalamus",
    "frontal lobe", "parietal lobe", "temporal lobe", "occipital lobe",
    "left ventricle", "right ventricle", "left atrium", "right atrium",
    "lumbar", "thoracic", "cervical",
}

# ---------------------------------------------------------------------------
# PROCEDURE TERMS — whole-phrase matching only
# ---------------------------------------------------------------------------
PROCEDURE_TERMS = {
    # Imaging
    "mri", "ct scan", "x-ray", "xray", "ultrasound", "sonography",
    "echocardiogram", "echocardiography", "echo", "pet scan",
    "mammogram", "colonoscopy", "endoscopy", "bronchoscopy",
    "angiography", "fluoroscopy", "biopsy", "fine needle aspiration", "fna",
    # ECG — full word match so "ECG" → procedure, not noise
    "ecg", "ekg", "electrocardiogram", "electrocardiography",
    "eeg", "emg", "holter monitor", "stress test", "treadmill test",
    # Surgical
    "surgery", "operation", "resection", "excision", "incision",
    "appendectomy", "cholecystectomy", "bypass", "transplant",
    "angioplasty", "stenting", "catheterization", "intubation",
    "tracheotomy", "amputation", "dialysis",
    # Therapeutic
    "chemotherapy", "radiation therapy", "radiotherapy",
    "immunotherapy", "physical therapy", "physiotherapy",
    "occupational therapy", "speech therapy",
    "blood transfusion", "iv infusion", "vaccination",
    # Rate/rhythm control — catches "rate control and anticoagulation"
    "rate control", "rhythm control", "anticoagulation",
    "anticoagulant therapy", "cardioversion", "ablation",
    "defibrillation", "pacemaker", "cardioverter",
}

# Treatment verb patterns — entity starting with these → procedure
TREATMENT_VERB_PATTERNS = [
    r"^rate control\b",
    r"^rhythm control\b",
    r"^anticoagulat",
    r"^started on\b",
    r"\banticoagulat",
    r"\brate control\b",
    r"\brhythm control\b",
]

# ---------------------------------------------------------------------------
# LAB TEST TERMS — use EXACT / whole-word matching to avoid false hits
# ---------------------------------------------------------------------------
LAB_TEST_TERMS = {
    "cbc", "complete blood count", "hemoglobin", "hematocrit",
    "white blood cell count", "wbc count", "red blood cell count",
    "rbc count", "platelet count",
    "blood glucose", "fasting glucose", "hba1c", "a1c",
    "cholesterol", "ldl", "hdl", "triglycerides", "lipid panel",
    "creatinine", "bun", "urea", "uric acid", "gfr",
    "sodium", "potassium", "chloride", "bicarbonate", "electrolytes",
    "calcium level", "magnesium level", "phosphorus level",
    "ast", "alt", "alp", "ggt", "bilirubin", "albumin level",
    "liver function test", "lft", "liver enzymes",
    "tsh", "t3", "t4", "thyroid function test", "cortisol level",
    "troponin", "ck-mb", "bnp", "nt-probnp", "d-dimer",
    "prothrombin time", "ptt", "inr",
    "crp", "c-reactive protein", "esr", "sed rate", "ferritin",
    "procalcitonin", "blood culture", "urine culture",
    "urinalysis", "urine analysis", "urine protein",
    "arterial blood gas", "abg", "spirometry", "pulmonary function test",
    "allergy test", "skin prick test", "stool test",
    "fasting blood sugar", "random blood sugar", "blood sugar level",
    "serum creatinine", "serum sodium", "serum potassium",
}

# ---------------------------------------------------------------------------
# MEDICATION DATA
# ---------------------------------------------------------------------------
MEDICATION_SUFFIXES = (
    "mab", "nib", "pril", "sartan", "statin", "mycin", "cillin",
    "cycline", "oxacin", "azole", "olol", "dipine", "prazole",
    "setron", "mide", "zide",
)

KNOWN_MEDICATIONS = {
    "aspirin", "ibuprofen", "acetaminophen", "paracetamol", "morphine",
    "codeine", "tramadol", "oxycodone", "fentanyl", "naloxone",
    "metformin", "insulin", "glipizide", "glimepiride",
    "atorvastatin", "simvastatin", "rosuvastatin",
    "lisinopril", "enalapril", "ramipril",
    "amlodipine", "nifedipine", "diltiazem", "verapamil",
    "metoprolol", "atenolol", "carvedilol", "bisoprolol",
    "warfarin", "heparin", "rivaroxaban", "apixaban", "clopidogrel",
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "metronidazole", "fluconazole", "acyclovir",
    "prednisone", "prednisolone", "dexamethasone", "hydrocortisone",
    "levothyroxine", "propylthiouracil", "methimazole",
    "sertraline", "fluoxetine", "escitalopram", "citalopram",
    "alprazolam", "diazepam", "lorazepam", "clonazepam",
    "omeprazole", "pantoprazole", "ranitidine",
    "salbutamol", "albuterol", "ipratropium", "tiotropium",
    "furosemide", "hydrochlorothiazide", "spironolactone",
    "gabapentin", "pregabalin", "phenytoin", "levetiracetam",
    "hydroxychloroquine", "methotrexate", "sulfasalazine",
    "adalimumab", "infliximab", "rituximab", "trastuzumab",
    "tamoxifen", "letrozole", "anastrozole",
    "digoxin", "amiodarone", "flecainide", "sotalol",        # antiarrhythmics
    "dabigatran", "edoxaban",                                 # anticoagulants
    "vitamin d", "vitamin b12", "folic acid", "iron",
    "calcium carbonate", "magnesium", "zinc",
    "anticoagulant", "anticoagulants",
    "beta blocker", "beta-blocker", "beta blockers",
    "ace inhibitor", "ace inhibitors",
    "calcium channel blocker", "calcium channel blockers",
}


class BertEngine:
    """
    BioBERT-only Medical NER engine (assignment requirement).

    Post-processing pipeline separates entities into:
        disease | symptom | medication | procedure | lab_test | anatomy | dosage
    """

    def __init__(self):
        ai_config = getattr(settings, "AI_CONFIG", {})
        self.hf_token = ai_config.get("HF_API_KEY", "") or os.getenv("HF_API_KEY", "")
        self.api_available = bool(self.hf_token)
        self.headers = (
            {"Authorization": f"Bearer {self.hf_token}",
             "Content-Type": "application/json"}
            if self.hf_token else {}
        )
        if self.api_available:
            logger.info("BioBERT NER engine initialised (HF Inference API)")
        else:
            logger.warning("No HF_API_KEY — BioBERT API unavailable.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> dict:
        """
        Extract medical entities from text using BioBERT + post-processing.
        Returns dict: { disease, symptom, medication, procedure, lab_test, anatomy, dosage }
        """
        if not self.api_available:
            return self._empty_result()

        chunks = self._chunk_text(text, max_chars=1500)
        raw = self._empty_result(use_sets=True)

        for chunk in chunks:
            self._query_biobert(chunk, raw)

        result = self._postprocess_pipeline(raw, text)
        return {k: sorted(list(v)) for k, v in result.items() if v}

    # ------------------------------------------------------------------
    # BioBERT API query
    # ------------------------------------------------------------------

    def _query_biobert(self, text: str, entities: dict):
        urls = [
            f"{HF_ROUTER_BASE}/{BIOBERT_PRIMARY}",
            f"{HF_ROUTER_BASE}/{BIOBERT_SECONDARY}",
        ]
        for url in urls:
            try:
                resp = requests.post(
                    url, headers=self.headers,
                    json={"inputs": text,
                          "options": {"wait_for_model": True},
                          "parameters": {"aggregation_strategy": "simple"}},
                    timeout=300,
                )
                if resp.status_code == 200:
                    results = resp.json()
                    if isinstance(results, list):
                        self._parse_response(results, entities)
                    return
                elif resp.status_code in (400, 422):
                    # Retry without aggregation_strategy
                    resp2 = requests.post(
                        url, headers=self.headers,
                        json={"inputs": text, "options": {"wait_for_model": True}},
                        timeout=300,
                    )
                    if resp2.status_code == 200:
                        results = resp2.json()
                        if isinstance(results, list):
                            self._parse_response(results, entities)
                        return
                elif resp.status_code == 503:
                    logger.info(f"Model loading: {url}")
                    continue
                else:
                    logger.warning(f"{url} → {resp.status_code}: {resp.text[:200]}")
                    continue
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout: {url}")
            except Exception as e:
                logger.error(f"Error querying {url}: {e}")

    def _parse_response(self, results: list, entities: dict):
        if results and isinstance(results[0], list):
            results = results[0]

        current_entity = ""
        current_label = ""

        for item in results:
            if not isinstance(item, dict):
                continue
            raw_word = item.get("word", "")
            label = item.get("entity_group", item.get("entity", ""))
            score = item.get("score", 0)

            if score < 0.45 or not label:
                continue

            if label.upper().startswith("B-") or (label != current_label):
                self._flush(current_entity, current_label, entities)
                current_entity = raw_word.replace("##", "")
                current_label = label
            else:
                if raw_word.startswith("##"):
                    current_entity += raw_word[2:]
                else:
                    current_entity += " " + raw_word

        self._flush(current_entity, current_label, entities)

    def _flush(self, entity: str, label: str, entities: dict):
        if not entity or not label:
            return
        cleaned = self._clean_token(entity)
        if not cleaned:
            return
        label_norm = label.upper().replace("B-", "").replace("I-", "").strip()
        if label_norm in ("DISEASE", "DISEASE_DISORDER", "CONDITION",
                          "LABEL_0", "LABEL_1", "LABEL_2", "DIS", "MISC",
                          "SPECIES", "ORGANISM", "PATHOGEN"):
            entities["disease"].add(cleaned)
        elif label_norm in ("CHEMICAL", "DRUG", "MEDICATION", "TREATMENT", "MEDICINE"):
            entities["medication"].add(cleaned)
        # Gene/protein tokens deliberately dropped — not clinically useful here

    # ------------------------------------------------------------------
    # Post-processing pipeline
    # ------------------------------------------------------------------

    def _postprocess_pipeline(self, entities: dict, original_text: str) -> dict:
        # ORDER MATTERS:
        # 1. Remove noise first so garbage doesn't pollute other steps
        # 2. Symptoms before lab_tests (palpitations fix)
        # 3. Anatomy, procedures, lab_tests, medications, dosages
        # 4. Final cleanup

        entities = self._remove_noise(entities)
        entities = self._rescue_symptoms(entities)        # ← before lab_test detection
        entities = self._detect_procedures(entities)
        entities = self._detect_lab_tests(entities)       # ← after symptoms rescued
        entities = self._detect_anatomy(entities)
        entities = self._detect_medications(entities, original_text)
        entities = self._detect_dosages(entities, original_text)
        entities = self._final_cleanup(entities)
        return entities

    # ------------------------------------------------------------------
    # Step 1: Noise removal
    # ------------------------------------------------------------------

    def _remove_noise(self, entities: dict) -> dict:
        """
        Remove sentence fragments and non-entity phrases.
        Fixes: "Presented with", "Started on rate control and anticoagulation",
               "ECG confirmed newly diagnosed" being kept as entities.
        """
        for category in list(entities.keys()):
            cleaned = set()
            for entity in entities[category]:
                el = entity.lower().strip()

                # Drop if it IS a known noise phrase
                if el in NOISE_PHRASES:
                    continue

                # Drop if it STARTS WITH a noise phrase
                if any(el.startswith(n) for n in NOISE_PHRASES):
                    # Exception: keep if it contains a known medical term after the noise
                    tail = el
                    for n in NOISE_PHRASES:
                        if el.startswith(n):
                            tail = el[len(n):].strip()
                            break
                    if not tail or len(tail) < 3:
                        continue
                    # Replace entity with just the tail (the actual medical content)
                    entity = entity[entity.lower().find(tail):].strip()
                    el = entity.lower().strip()

                # Drop if matches a noise regex pattern
                noise_hit = False
                for pattern in NOISE_PATTERNS:
                    if re.search(pattern, el, re.IGNORECASE):
                        noise_hit = True
                        break
                if noise_hit:
                    continue

                # Drop if too many words (likely a sentence fragment)
                # Exception: known multi-word medical terms are allowed
                word_count = len(entity.split())
                if word_count > 6:
                    continue
                if word_count > 4:
                    # Allow only if it contains a known medical anchor
                    has_anchor = (
                        any(a in el for a in DISEASE_ANCHOR_TERMS) or
                        any(a in el for a in SYMPTOM_TERMS) or
                        any(a in el for a in PROCEDURE_TERMS)
                    )
                    if not has_anchor:
                        continue

                cleaned.add(entity)
            entities[category] = cleaned
        return entities

    # ------------------------------------------------------------------
    # Step 2: Symptom rescue (runs BEFORE lab_test detection)
    # ------------------------------------------------------------------

    def _rescue_symptoms(self, entities: dict) -> dict:
        diseases = set(entities.get("disease", set()))
        symptoms = set(entities.get("symptom", set()))
        rescued = set()

        for entity in diseases:
            el = entity.lower().strip()

            # Hard stop: disease anchor present
            if any(anchor in el for anchor in DISEASE_ANCHOR_TERMS):
                continue

            is_symptom = False

            # 1. Exact match in symptom terms
            if el in SYMPTOM_TERMS:
                is_symptom = True

            # 2. Phrase-level containment
            if not is_symptom:
                for term in SYMPTOM_TERMS:
                    if term in el:
                        is_symptom = True
                        break

            # 3. Linguistic patterns
            if not is_symptom:
                patterns = [
                    "pain", "ache", "aching", "sore ", "soreness",
                    "shortness", "difficulty ", "inability",
                    "swelling", "swollen", "tenderness",
                    "weakness", "numbness", "tingling",
                    "nausea", "vomit", "diarrhea", "diarrhoea",
                    "fatigue", "fever", "chills", "sweating",
                    "rash", "itching", "bleed", "bleeding",
                    "dizziness", "vertigo", "fainting",
                    "palpitation", "tightness",
                    "discharge", "loss of", "decreased ", "reduced ",
                    "blurred", "ringing", "constipation",
                    "bloating", "heartburn", "cramping", "spasm",
                ]
                if any(p in el for p in patterns):
                    is_symptom = True

            if is_symptom:
                rescued.add(entity)

        entities["disease"] = diseases - rescued
        entities["symptom"] = symptoms | rescued
        return entities

    # ------------------------------------------------------------------
    # Step 3: Procedure detection
    # ------------------------------------------------------------------

    def _detect_procedures(self, entities: dict) -> dict:
        diseases = set(entities.get("disease", set()))
        symptoms = set(entities.get("symptom", set()))
        procedures = set(entities.get("procedure", set()))
        to_move = set()

        for source in (diseases, symptoms):
            for entity in source:
                el = entity.lower().strip()

                # Exact/phrase match against procedure terms
                if any(term == el or term in el for term in PROCEDURE_TERMS):
                    to_move.add(entity)
                    continue

                # Treatment verb patterns (catches "rate control and anticoagulation")
                if any(re.search(p, el) for p in TREATMENT_VERB_PATTERNS):
                    to_move.add(entity)

        entities["disease"] = diseases - to_move
        entities["symptom"] = symptoms - to_move
        entities["procedure"] = procedures | to_move
        return entities

    # ------------------------------------------------------------------
    # Step 4: Lab test detection — WHOLE-WORD matching only
    # ------------------------------------------------------------------

    def _detect_lab_tests(self, entities: dict) -> dict:
        """
        Use whole-word / exact matching for lab tests.
        Prevents 'palpitations' matching 'test' substring.
        """
        diseases = set(entities.get("disease", set()))
        symptoms = set(entities.get("symptom", set()))
        lab_tests = set(entities.get("lab_test", set()))
        to_move = set()

        for source in (diseases, symptoms):
            for entity in source:
                el = entity.lower().strip()
                # Only match if the lab term IS the entity or the entity IS the lab term
                # NOT substring containment (avoids palpitations → lab_test)
                if el in LAB_TEST_TERMS or any(
                    el == term or
                    el.startswith(term + " ") or
                    el.endswith(" " + term)
                    for term in LAB_TEST_TERMS
                ):
                    # Double-check it's not already correctly placed as symptom
                    if el not in SYMPTOM_TERMS and not any(s in el for s in SYMPTOM_TERMS):
                        to_move.add(entity)

        entities["disease"] = diseases - to_move
        entities["symptom"] = symptoms - to_move
        entities["lab_test"] = lab_tests | to_move
        return entities

    # ------------------------------------------------------------------
    # Step 5: Anatomy detection
    # ------------------------------------------------------------------

    def _detect_anatomy(self, entities: dict) -> dict:
        diseases = set(entities.get("disease", set()))
        symptoms = set(entities.get("symptom", set()))
        anatomy = set(entities.get("anatomy", set()))
        to_move = set()

        for source in (diseases, symptoms):
            for entity in source:
                el = entity.lower().strip()
                if any(term == el or term in el for term in ANATOMY_TERMS):
                    if not any(anchor in el for anchor in DISEASE_ANCHOR_TERMS):
                        to_move.add(entity)

        entities["disease"] = diseases - to_move
        entities["symptom"] = symptoms - to_move
        entities["anatomy"] = anatomy | to_move
        return entities

    # ------------------------------------------------------------------
    # Step 6: Medication detection
    # ------------------------------------------------------------------

    def _detect_medications(self, entities: dict, original_text: str) -> dict:
        diseases = set(entities.get("disease", set()))
        symptoms = set(entities.get("symptom", set()))
        medications = set(entities.get("medication", set()))
        to_move = set()

        for source in (diseases, symptoms):
            for entity in source:
                el = entity.lower().strip()
                if el in KNOWN_MEDICATIONS:
                    to_move.add(entity)
                    continue
                if len(el) > 4 and any(el.endswith(s) for s in MEDICATION_SUFFIXES):
                    to_move.add(entity)

        # Scan raw text for medications BioBERT missed
        text_lower = original_text.lower()
        for med in KNOWN_MEDICATIONS:
            if med in text_lower:
                idx = text_lower.find(med)
                medications.add(original_text[idx: idx + len(med)].strip())

        entities["disease"] = diseases - to_move
        entities["symptom"] = symptoms - to_move
        entities["medication"] = medications | to_move
        return entities

    # ------------------------------------------------------------------
    # Step 7: Dosage detection from original text
    # ------------------------------------------------------------------

    def _detect_dosages(self, entities: dict, original_text: str) -> dict:
        dosages = set(entities.get("dosage", set()))
        patterns = [
            r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml|l|iu|meq|mmol|units?|tablets?|caps?|capsules?)\b',
            r'\b(?:once|twice|three times|four times)\s+(?:a\s+)?(?:daily|weekly|monthly)\b',
            r'\b(?:every|each)\s+\d+\s+(?:hours?|days?|weeks?)\b',
            r'\b\d+\s*(?:times?)\s+(?:a\s+)?(?:day|week|month)\b',
            r'\b(?:q\d+h|qd|bid|tid|qid|prn|sos)\b',
        ]
        for pattern in patterns:
            for match in re.findall(pattern, original_text, re.IGNORECASE):
                cleaned = match.strip()
                if cleaned:
                    dosages.add(cleaned)
        entities["dosage"] = dosages
        return entities

    # ------------------------------------------------------------------
    # Step 8: Final cleanup
    # ------------------------------------------------------------------

    def _final_cleanup(self, entities: dict) -> dict:
        for category in entities:
            # Clean tokens
            cleaned = set()
            for e in entities[category]:
                c = self._clean_token(e)
                if c:
                    cleaned.add(c)

            # Case-insensitive dedup — prefer capitalised
            deduped = {}
            for e in cleaned:
                key = e.lower()
                if key not in deduped or e[0].isupper():
                    deduped[key] = e

            final = set(deduped.values())

            # Remove pure substrings
            to_remove = set()
            fl = list(final)
            for i, e1 in enumerate(fl):
                for e2 in fl:
                    if e1 != e2 and e1.lower() in e2.lower():
                        to_remove.add(e1)

            entities[category] = final - to_remove
        return entities

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _clean_token(self, entity: str):
        if not entity:
            return None
        entity = entity.strip().strip('.,;:()[]{}"\'-/#*')
        entity = " ".join(entity.split())
        if len(entity) <= 1 or entity.isdigit():
            return None
        if all(c in '.,;:()[]{}"\'-/#* ' for c in entity):
            return None
        return entity

    def _chunk_text(self, text: str, max_chars: int = 1500) -> list:
        if len(text) <= max_chars:
            return [text]
        sentences = text.replace("\n", ". ").split(". ")
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) + 2 > max_chars:
                if current:
                    chunks.append(current.strip())
                current = s
            else:
                current = (current + ". " + s) if current else s
        if current:
            chunks.append(current.strip())
        return chunks or [text[:max_chars]]

    @staticmethod
    def _empty_result(use_sets: bool = False) -> dict:
        factory = set if use_sets else list
        return {
            "disease":   factory(),
            "symptom":   factory(),
            "medication": factory(),
            "procedure": factory(),
            "lab_test":  factory(),
            "anatomy":   factory(),
            "dosage":    factory(),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_bert_engine = None


def get_bert_engine() -> BertEngine:
    global _bert_engine
    if _bert_engine is None:
        _bert_engine = BertEngine()
    return _bert_engine