"""
Enhanced GraphRAG Engine - Knowledge Graph-based Retrieval Augmented Generation.
Builds and queries a medical knowledge graph from a comprehensive JSON knowledge base
and uses Vector-based semantic search for robust symptom matching.
"""
import logging
import json
import os
import re
from collections import defaultdict
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

# Try to import sentence_transformers for semantic matching
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence_transformers not installed. Falling back to exact symptom matching.")


class GraphRAGEngine:
    """
    Enhanced GraphRAG engine that builds a large medical knowledge graph from JSON
    and retrieves relevant context using vector-based semantic similarity and graph traversal.
    """

    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(list))
        self.kb_data = {}
        self.symptom_embeddings = None
        self.symptom_list = []
        self.model = None
        
        self._load_knowledge_base()
        self._build_knowledge_graph()
        self._init_vector_search()

    def _load_knowledge_base(self):
        """Load the comprehensive medical knowledge base from JSON."""
        # Ensure we look in the right place relative to this file
        base_dir = getattr(settings, 'BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        kb_path = os.path.join(base_dir, 'data', 'medical_knowledge_base.json')
        
        try:
            with open(kb_path, 'r') as f:
                self.kb_data = json.load(f)
            logger.info(f"Loaded medical knowledge base version {self.kb_data.get('metadata', {}).get('version', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to load knowledge base from {kb_path}: {e}")
            # Fallback to minimal dict to prevent crashes if file missing
            self.kb_data = {
                'disease_symptoms': {}, 'disease_treatments': {}, 'drug_interactions': {}, 
                'symptom_synonyms': {}
            }

    def _build_knowledge_graph(self):
        """Build knowledge graph from loaded ontology."""
        self.graph.clear()
        
        # Disease -> Symptom edges
        for disease, symptoms in self.kb_data.get('disease_symptoms', {}).items():
            for symptom in symptoms:
                self.graph[disease]['has_symptom'].append(symptom)
                self.graph[symptom]['symptom_of'].append(disease)

        # Disease -> Treatment edges
        for disease, treatments in self.kb_data.get('disease_treatments', {}).items():
            for treatment in treatments:
                self.graph[disease]['treated_by'].append(treatment)
                self.graph[treatment]['treats'].append(disease)

        # Drug interaction edges
        for drug, interactions in self.kb_data.get('drug_interactions', {}).items():
            for interacting in interactions:
                self.graph[drug]['interacts_with'].append(interacting)
                self.graph[interacting]['interacts_with'].append(drug)

        # Symptom synonyms mapping (maps synonym back to canonical symptom)
        for canonical, synonyms in self.kb_data.get('symptom_synonyms', {}).items():
            for syn in synonyms:
                self.graph[syn]['canonical_symptom'] = canonical

        logger.info(f"Knowledge graph built: {len(self.graph)} nodes")

    def _init_vector_search(self):
        """Initialize sentence-transformers model for semantic symptom matching."""
        if not HAS_SENTENCE_TRANSFORMERS:
            return
            
        try:
            # all-MiniLM-L6-v2 is extremely fast, lightweight (80MB), and good for semantic matching
            logger.info("Loading SentenceTransformer model for semantic matching...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Extract all canonical symptoms from the graph
            self.symptom_list = list(self.kb_data.get('disease_symptoms', {}).values())
            # Flatten list of lists
            self.symptom_list = list(set([item for sublist in self.symptom_list for item in sublist]))
            
            # Include synonyms in the searchable list for better matching
            for synonyms in self.kb_data.get('symptom_synonyms', {}).values():
                self.symptom_list.extend(synonyms)
                
            self.symptom_list = list(set(self.symptom_list))
            
            if self.symptom_list:
                self.symptom_embeddings = self.model.encode(self.symptom_list, convert_to_tensor=True)
                logger.info(f"Vector search initialized for {len(self.symptom_list)} symptoms/synonyms")
        except Exception as e:
            logger.error(f"Failed to initialize sentence_transformers: {e}")
            self.model = None

    def retrieve_context(self, entities):
        """Retrieve relevant context from the knowledge graph given extracted entities."""
        context = {
            'related_diseases': [],
            'suggested_treatments': [],
            'drug_interactions': [],
            'symptom_disease_mapping': [],
            'emergency_flags': [],
            'risk_factors_matched': [],
        }

        # 1. Expand and match symptoms (using Vector Search if available, fallback to fuzzy)
        matched_canonical_symptoms = self._match_symptoms(entities.get('symptom', []))
        
        # Add extracted diseases directly as strong signals
        extracted_diseases = [d.lower() for d in entities.get('disease', [])]

        # 2. Find diseases matching symptoms
        symptom_matches = defaultdict(int)
        
        # Add points for matched symptoms
        for symptom in matched_canonical_symptoms:
            diseases = self.graph.get(symptom, {}).get('symptom_of', [])
            if not diseases and symptom in self.graph and 'canonical_symptom' in self.graph[symptom]:
                canon = self.graph[symptom]['canonical_symptom']
                diseases = self.graph.get(canon, {}).get('symptom_of', [])
                
            for disease in diseases:
                symptom_matches[disease] += 1
                
            if diseases:
                context['symptom_disease_mapping'].append({
                    'symptom': symptom, 'possible_diseases': diseases[:3]  # Only top 3 for brevity
                })

        # Add huge bonus points if the disease itself was explicitly extracted by NER
        for disease in extracted_diseases:
            # Fuzzy match to our KB diseases
            for kb_disease in self.kb_data.get('disease_symptoms', {}).keys():
                if kb_disease in disease or disease in kb_disease:
                    symptom_matches[kb_disease] += 5  # Strong weight for explicit disease mention

        # Rank diseases by match score
        for disease, score in sorted(symptom_matches.items(), key=lambda x: -x[1]):
            actual_symptoms = self.kb_data.get('disease_symptoms', {}).get(disease, [])
            total_symptoms = len(actual_symptoms)
            
            # Base confidence on how many symptoms matched out of typical presentation
            # (score can exceed total_symptoms due to explicit disease mentions)
            confidence = min(score / max(total_symptoms, 1), 1.0)
            
            # If explicitly extracted, boost confidence
            if any(disease in ext_d or ext_d in disease for ext_d in extracted_diseases):
                confidence = max(confidence, 0.9)
                
            context['related_diseases'].append({
                'disease': disease,
                'matching_score': score,
                'total_symptoms_in_kb': total_symptoms,
                'confidence': round(confidence, 2),
                'department': self.kb_data.get('disease_departments', {}).get(disease, 'General Medicine')
            })

        # Check for emergency conditions among top diseases
        top_diseases = [d['disease'] for d in context['related_diseases'][:3]]
        emergency_list = self.kb_data.get('emergency_conditions', [])
        for d in top_diseases:
            if d in emergency_list:
                context['emergency_flags'].append(f"URGENT: {d} is a medical emergency requiring immediate evaluation.")

        # 3. Get treatments for top 3 diseases
        for d_info in context['related_diseases'][:3]:
            disease = d_info['disease']
            treatments = self.kb_data.get('disease_treatments', {}).get(disease, [])
            if treatments:
                context['suggested_treatments'].extend([
                    {'treatment': t, 'for_disease': disease} for t in treatments[:5]  # Limit to top 5
                ])

        # 4. Check drug interactions
        medications = [e.lower() for e in entities.get('medication', [])]
        for i, med1 in enumerate(medications):
            # Find interactions in our KB
            node = self.graph.get(med1)
            
            # Try fuzzy match if exact fails
            if not node:
                for key in self.kb_data.get('drug_interactions', {}).keys():
                    if med1 in key or key in med1:
                        node = self.graph[key]
                        break
                        
            if node:
                for interacting in node.get('interacts_with', []):
                    # Check if the interacting drug is also in the patient's list
                    for med2 in medications[i+1:]:
                        if interacting in med2 or med2 in interacting:
                            context['drug_interactions'].append({
                                'drug1': med1, 'drug2': med2,
                                'warning': f"Potential interaction between {med1} and {med2}"
                            })

        return context

    def _match_symptoms(self, extracted_symptoms):
        """Map extracted raw symptoms to canonical KB symptoms using Vectors or Fuzzy matching."""
        matched = set()
        
        if not extracted_symptoms:
            return list(matched)
            
        # 1. Semantic Vector Search (Fast and highly accurate)
        if hasattr(self, 'model') and self.model is not None and self.symptom_embeddings is not None:
            try:
                queries = [s.lower() for s in extracted_symptoms]
                query_embeddings = self.model.encode(queries, convert_to_tensor=True)
                
                # Compute cosine similarities
                cosine_scores = util.cos_sim(query_embeddings, self.symptom_embeddings)
                
                # Find best matches above a threshold
                for i in range(len(queries)):
                    best_idx = np.argmax(cosine_scores[i].cpu().numpy())
                    best_score = cosine_scores[i][best_idx].item()
                    
                    if best_score > 0.65:  # Good semantic match threshold
                        matched_symptom = self.symptom_list[best_idx]
                        
                        # Resolve synonym to canonical
                        if matched_symptom in self.graph and 'canonical_symptom' in self.graph[matched_symptom]:
                            matched.add(self.graph[matched_symptom]['canonical_symptom'])
                        else:
                            matched.add(matched_symptom)
                        
                        logger.debug(f"Vector Match: '{queries[i]}' -> '{matched_symptom}' (Score: {best_score:.2f})")
                        continue
            except Exception as e:
                logger.warning(f"Vector search failed during matching: {e}")
                # Fall through to fuzzy matching

        # 2. Add Exact/Fuzzy Matching (Always run for exact matches, or as fallback)
        for entity in extracted_symptoms:
            entity_lower = entity.lower()
            
            # Exact match
            if entity_lower in self.graph and 'symptom_of' in self.graph[entity_lower]:
                matched.add(entity_lower)
                continue
                
            # Synonym exact match
            if entity_lower in self.graph and 'canonical_symptom' in self.graph[entity_lower]:
                matched.add(self.graph[entity_lower]['canonical_symptom'])
                continue
            
            # Basic fuzzy match against canonicals
            for kb_symptom in set(s for symptoms in self.kb_data.get('disease_symptoms', {}).values() for s in symptoms):
                if entity_lower in kb_symptom or kb_symptom in entity_lower:
                    matched.add(kb_symptom)

        return list(matched)

    def get_graph_stats(self):
        """Return graph statistics."""
        node_count = len(self.graph)
        edge_count = sum(
            len(targets) for node in self.graph.values() for targets in node.values()
        )
        return {
            'nodes': node_count, 
            'edges': edge_count,
            'diseases_in_kb': len(self.kb_data.get('disease_symptoms', {})),
            'vector_search_enabled': HAS_SENTENCE_TRANSFORMERS and self.model is not None
        }


# Singleton
_graph_engine = None

def get_graph_engine():
    global _graph_engine
    if _graph_engine is None:
        _graph_engine = GraphRAGEngine()
    return _graph_engine
