from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import PatientViewSet, DischargeSummaryViewSet, DiagnosisViewSet, KnowledgeGraphViewSet

router = DefaultRouter()
router.register(r'patients', PatientViewSet)
router.register(r'discharge-summaries', DischargeSummaryViewSet)
router.register(r'diagnoses', DiagnosisViewSet)
router.register(r'knowledge-graph', KnowledgeGraphViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
