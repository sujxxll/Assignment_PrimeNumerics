from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Patient, DischargeSummary, Diagnosis, KnowledgeGraphNode, KnowledgeGraphEdge
from .serializers import (
    PatientListSerializer, PatientDetailSerializer,
    DischargeSummarySerializer, DiagnosisSerializer, DiagnosisListSerializer,
    KnowledgeGraphNodeSerializer, KnowledgeGraphEdgeSerializer
)


class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all()

    def get_serializer_class(self):
        if self.action == 'list':
            return PatientListSerializer
        return PatientDetailSerializer

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Dashboard statistics."""
        total_patients = Patient.objects.count()
        total_diagnoses = Diagnosis.objects.count()
        total_summaries = DischargeSummary.objects.count()
        # Find how many summaries do NOT have an associated diagnosis yet
        pending = DischargeSummary.objects.filter(diagnoses__isnull=True).count()

        avg_confidence = 0
        if total_diagnoses > 0:
            from django.db.models import Avg
            avg_confidence = Diagnosis.objects.aggregate(avg=Avg('diagnosis_confidence'))['avg'] or 0

        # Department distribution
        from django.db.models import Count
        departments = list(
            Patient.objects.values('department')
            .annotate(count=Count('id'))
            .order_by('-count')
        )

        # Recent diagnoses
        recent = DiagnosisListSerializer(
            Diagnosis.objects.select_related('patient').order_by('-created_at')[:5],
            many=True
        ).data

        return Response({
            'total_patients': total_patients,
            'total_diagnoses': total_diagnoses,
            'total_summaries': total_summaries,
            'pending_diagnoses': pending,
            'average_confidence': round(avg_confidence, 2),
            'departments': departments,
            'recent_diagnoses': recent,
        })


class DischargeSummaryViewSet(viewsets.ModelViewSet):
    queryset = DischargeSummary.objects.select_related('patient').all()
    serializer_class = DischargeSummarySerializer


class DiagnosisViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Diagnosis.objects.select_related('patient', 'discharge_summary').all()

    def get_serializer_class(self):
        if self.action == 'list':
            return DiagnosisListSerializer
        return DiagnosisSerializer


class KnowledgeGraphViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = KnowledgeGraphNode.objects.all()
    serializer_class = KnowledgeGraphNodeSerializer

    @action(detail=False, methods=['get'])
    def graph_data(self, request):
        """Return full graph data for visualization."""
        nodes = KnowledgeGraphNodeSerializer(
            KnowledgeGraphNode.objects.all()[:200], many=True
        ).data
        edges = KnowledgeGraphEdgeSerializer(
            KnowledgeGraphEdge.objects.select_related('source', 'target').all()[:500], many=True
        ).data
        return Response({'nodes': nodes, 'edges': edges})

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Graph statistics."""
        from django.db.models import Count
        node_types = list(
            KnowledgeGraphNode.objects.values('node_type')
            .annotate(count=Count('id'))
            .order_by('-count')
        )
        total_edges = KnowledgeGraphEdge.objects.count()
        return Response({
            'total_nodes': KnowledgeGraphNode.objects.count(),
            'total_edges': total_edges,
            'node_types': node_types,
        })
