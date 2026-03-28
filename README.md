# HealthAI - Intelligent Clinical Diagnosis System

An advanced, end-to-end healthcare AI system that processes raw patient discharge summaries through a highly-optimized **BioBERT → Enhanced Vector GraphRAG → Med42 LLM Engine** pipeline. It extracts critical medical entities, maps them against a vast, semantically-searchable medical ontology, and uses Large Language Models to generate actionable, structured clinical diagnoses and treatment plans.

## 🌟 Key Features

- **Real-Time Advanced AI Pipeline**: Operates locally using efficient HuggingFace endpoints without expensive OpenAI dependencies.
- **Enhanced Vector GraphRAG**: Processes fuzzy medical symptom descriptions using `sentence-transformers` vector embeddings to seamlessly map patient notes onto a pristine 120-disease clinical Knowledge Base.
- **Dynamic Database Architecture**: Switch securely between local `SQLite` (for dev) and `MySQL` (for production) using a simple `.env` toggle (`USE_SQLITE=true|false`).
- **Comprehensive Logging Engine**: Fully centralized system capturing all stack-traces, timeouts, and API disruptions into a persistent `backend/logs/application_errors.log` file.
- **VAPT Secured**: Production-level Django deployment security policies dynamically inject secure HTTP headers, session constraints, and XSS filtering whenever `DEBUG=False`. 
- **Modern User Interface**: A responsive, dark-themed dashboard built with Next.js 16 and Material UI tracking pipeline statistics and confidence metrics.

---

## 🧬 The Core AI Pipeline

```text
[Discharge Summary] 
        ↓ 
    BioBERT NER (Named Entity Extraction)
    Extracts: Symptoms, Diseases, Medications
        ↓ 
    Vector GraphRAG (Sentence-Transformers)
    Mathematical Cosine Similarity mapping to Clinical Ontology
        ↓
    Knowledge Context Retrieval 
    Fetches: Drug Interactions, Suggested Treatments, Emergency Flags
        ↓ 
    Med42 Clinical LLM Engine 
        ↓ 
    [Structured Primary/Secondary Diagnosis + Treatment Plan]
```

---

## 🛠 Tech Stack

- **Frontend**: Next.js 16.2.1, React 19, Material UI (MUI), Axios
- **Backend / API**: Django 6.0, Django REST Framework
- **Databases**: Adaptive SQLite / MySQL / PyMySQL patched
- **NLP & Named Entity Recognition**: BioBERT fine-tunes (`d4data/biomedical-ner-all`)
- **Semantic Vector Engine**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Clinical LLM**: Med42 (`m42-health/Llama3-Med42-8B`) via Featherless AI / HuggingFace
- **Ontology & Knowledge Base**: A massive local JSON store holding 120+ clinical diseases, treatments, interaction rules, and ICD-10 data.

---

## 🚀 Quick Start Guide

### 1. Backend Setup
Ensure you have Python 3.10+ installed.

```bash
cd backend
pip install -r requirements.txt
pip install sentence-transformers  # Required for Local Vector GraphRAG
```

Set up your `.env` file in the `backend/` directory:
```ini
USE_SQLITE=true
DEBUG=true
HF_API_KEY=hf_your_api_key_here
```

Apply migrations and start the server:
```bash
python manage.py migrate
python manage.py runserver 8000
```
> *Note: The backend will automatically create `/logs/application_errors.log` to track system health and execution traces securely.*

### 2. Frontend Setup
Ensure you have Node.js 18+ installed.

```bash
cd frontend
npm install
npm run dev
```
The dashboard will dynamically spin up and map to **`http://localhost:3000`**.

---

## 🗄️ Database Configurations

You can heavily customize where HealthAI stores the patient and diagnoses metrics via your `.env` configuration.

### Working with Local SQLite (Default Demo Mode)
For development without requiring a separate database server:
```ini
USE_SQLITE=true
```

### Full Production MySQL Setup
If you are running a local XAMPP/WAMP or Dockerized MySQL instance on port `3306`:
```ini
USE_SQLITE=false
DB_NAME=healthcare_ai
DB_USER=root
DB_PASSWORD=root
DB_HOST=127.0.0.1
DB_PORT=3306
```
> *Django will safely manage database drivers relying on the patched PyMySQL architecture.*

## 📊 Live API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/patients/` | GET | List all active patients |
| `/api/patients/stats/` | GET | Advanced pipeline dashboard statistics |
| `/api/diagnose/<patient_id>/` | POST | Trigger the full End-to-End LLM diagnosis for one patient |
| `/api/diagnose-all/` | POST | Bulk queue diagnoses for all missing/pending patients |
| `/api/diagnoses/` | GET | Fetch the LLM-generated structures |
| `/api/pipeline-status/` | GET | Inspect APIs, Model Connectors, and Embedding states |
| `/api/knowledge-graph/graph_data/` | GET | Retrieve localized raw Vector relationships |

---

## 📝 Error Tracking & Handling
HealthAI no longer times out unexpectedly or fails silently when external free-tier endpoints queue or cold-boot:
- **Resilient AI Calling:** The Python and Axios layers are properly shielded against unexpected API stalls (extended up to 300-second soft-limits).
- **Hard Logging:** All raw exceptions thrown during Django's `runserver` process, LLM JSON decoding failures, or database migration collisions are silently caught and logged chronologically in **`backend/logs/application_errors.log`**.
