"""
Diagnosis Pipeline - Orchestrates: BioBERT NER → GraphRAG → LLM Ensemble
"""
import logging
import time
from .bert_engine import get_bert_engine
from .graph_rag import get_graph_engine
from .llm_engine import get_llm_engine

logger = logging.getLogger(__name__)


class DiagnosisPipeline:
    """
    End-to-end diagnosis pipeline:
    1. BioBERT extracts medical entities
    2. GraphRAG retrieves relevant knowledge context
    3. LLM ensemble generates diagnosis and treatment plan
    """

    def __init__(self):
        self.bert = get_bert_engine()
        self.graph_rag = get_graph_engine()
        self.llm = get_llm_engine()

    def run(self, discharge_summary_text):
        """Execute the full diagnosis pipeline."""
        start = time.time()
        pipeline_log = []

        # Step 1: BERT Entity Extraction
        t1 = time.time()
        entities = self.bert.extract_entities(discharge_summary_text)
        pipeline_log.append({
            'step': 'BioBERT NER',
            'duration': round(time.time() - t1, 3),
            'entities_found': sum(len(v) for v in entities.values()),
        })
        logger.info(f"Step 1 - Entities extracted: {sum(len(v) for v in entities.values())}")

        # Step 2: GraphRAG Context Retrieval
        t2 = time.time()
        graph_context = self.graph_rag.retrieve_context(entities)
        pipeline_log.append({
            'step': 'GraphRAG Context Retrieval',
            'duration': round(time.time() - t2, 3),
            'related_diseases': len(graph_context.get('related_diseases', [])),
            'treatments_found': len(graph_context.get('suggested_treatments', [])),
        })
        logger.info(f"Step 2 - GraphRAG: {len(graph_context.get('related_diseases', []))} diseases matched")

        # Step 3: LLM Ensemble Diagnosis
        t3 = time.time()
        diagnosis = self.llm.generate_diagnosis(discharge_summary_text, entities, graph_context)
        pipeline_log.append({
            'step': 'LLM Inference (Med42)',
            'duration': round(time.time() - t3, 3),
            'models_used': diagnosis.get('models_used', []),
        })
        logger.info(f"Step 3 - LLM diagnosis generated: {diagnosis['primary_diagnosis'][:50]}")

        total_time = round(time.time() - start, 3)
        return {
            'entities': entities,
            'graph_context': graph_context,
            'diagnosis': diagnosis,
            'pipeline_log': pipeline_log,
            'total_processing_time': total_time,
        }


# Singleton
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = DiagnosisPipeline()
    return _pipeline
