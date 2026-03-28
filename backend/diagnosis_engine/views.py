from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from patients.models import Patient, DischargeSummary, Diagnosis
from patients.serializers import DiagnosisSerializer
from .pipeline import get_pipeline


class RunDiagnosisView(APIView):
    """Run the AI diagnosis pipeline for a specific patient."""

    def post(self, request, patient_id):
        try:
            patient = Patient.objects.get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return Response({'error': 'Patient not found'}, status=status.HTTP_404_NOT_FOUND)

        try:
            summary = patient.discharge_summary
        except DischargeSummary.DoesNotExist:
            return Response({'error': 'No discharge summary found'}, status=status.HTTP_404_NOT_FOUND)

        # Run pipeline
        try:
            pipeline = get_pipeline()
            result = pipeline.run(summary.full_summary_text)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing single patient {summary.patient.patient_id}", exc_info=True)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        dx = result['diagnosis']

        # Save to database
        diagnosis = Diagnosis.objects.create(
            patient=patient,
            discharge_summary=summary,
            extracted_entities=result['entities'],
            graph_context=result['graph_context'],
            primary_diagnosis=dx['primary_diagnosis'],
            secondary_diagnoses=dx.get('secondary_diagnoses', []),
            diagnosis_confidence=dx.get('diagnosis_confidence', 0),
            diagnosis_reasoning=dx.get('diagnosis_reasoning', ''),
            treatment_plan=dx.get('treatment_plan', ''),
            medications_recommended=dx.get('medications_recommended', []),
            procedures_recommended=dx.get('procedures_recommended', []),
            lifestyle_modifications=dx.get('lifestyle_modifications', []),
            follow_up_schedule=dx.get('follow_up_schedule', []),
            models_used=dx.get('models_used', []),
            processing_time_seconds=result['total_processing_time'],
        )

        return Response({
            'diagnosis': DiagnosisSerializer(diagnosis).data,
            'pipeline_log': result['pipeline_log'],
        }, status=status.HTTP_201_CREATED)


class RunAllDiagnosesView(APIView):
    """Run diagnosis for all patients with discharge summaries."""

    def post(self, request):
        summaries = DischargeSummary.objects.select_related('patient').all()
        pipeline = get_pipeline()
        results = []

        for summary in summaries:
            # Skip if already diagnosed (unless force=true)
            if not request.data.get('force') and summary.patient.diagnoses.exists():
                results.append({
                    'patient_id': summary.patient.patient_id,
                    'status': 'skipped',
                    'message': 'Already diagnosed',
                })
                continue

            try:
                result = pipeline.run(summary.full_summary_text)
                dx = result['diagnosis']

                diagnosis = Diagnosis.objects.create(
                    patient=summary.patient,
                    discharge_summary=summary,
                    extracted_entities=result['entities'],
                    graph_context=result['graph_context'],
                    primary_diagnosis=dx['primary_diagnosis'],
                    secondary_diagnoses=dx.get('secondary_diagnoses', []),
                    diagnosis_confidence=dx.get('diagnosis_confidence', 0),
                    diagnosis_reasoning=dx.get('diagnosis_reasoning', ''),
                    treatment_plan=dx.get('treatment_plan', ''),
                    medications_recommended=dx.get('medications_recommended', []),
                    procedures_recommended=dx.get('procedures_recommended', []),
                    lifestyle_modifications=dx.get('lifestyle_modifications', []),
                    follow_up_schedule=dx.get('follow_up_schedule', []),
                    models_used=dx.get('models_used', []),
                    processing_time_seconds=result['total_processing_time'],
                )

                results.append({
                    'patient_id': summary.patient.patient_id,
                    'status': 'completed',
                    'primary_diagnosis': dx['primary_diagnosis'],
                    'confidence': dx.get('diagnosis_confidence', 0),
                    'processing_time': result['total_processing_time'],
                })
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error processing patient {summary.patient.patient_id}", exc_info=True)
                
                results.append({
                    'patient_id': summary.patient.patient_id,
                    'status': 'error',
                    'message': str(e),
                })

        return Response({
            'total_processed': len([r for r in results if r['status'] == 'completed']),
            'total_skipped': len([r for r in results if r['status'] == 'skipped']),
            'results': results,
        })


class PipelineStatusView(APIView):
    """Get the current status of the AI pipeline components."""

    def get(self, request):
        from .bert_engine import get_bert_engine
        from .graph_rag import get_graph_engine
        from .llm_engine import get_llm_engine, LLMEngine

        bert = get_bert_engine()
        graph = get_graph_engine()
        llm = get_llm_engine()

        return Response({
            'bert_engine': {
                'status': 'active',
                'api_connected': bert.api_available,
                'model': 'HuggingFace API (BioBERT NER)' if bert.api_available else 'Rule-based NER (no HF_API_KEY)',
                'ner_model': bert.NER_MODEL,
            },
            'graph_rag': {
                'status': 'active',
                'stats': graph.get_graph_stats(),
            },
            'llm_engine': {
                'status': 'active',
                'models': LLMEngine.MODELS,
                'huggingface_connected': llm.has_hf,
                'simulated_mode': llm.use_simulated,
                'mode': 'API-based' if llm.has_hf and not llm.use_simulated
                        else 'Clinical Reasoning (set API keys & USE_SIMULATED_LLM=false)',
            },
            'pipeline': 'ready',
        })
