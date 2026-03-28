from django.urls import path
from .views import RunDiagnosisView, RunAllDiagnosesView, PipelineStatusView

urlpatterns = [
    path('diagnose/<str:patient_id>/', RunDiagnosisView.as_view(), name='run-diagnosis'),
    path('diagnose-all/', RunAllDiagnosesView.as_view(), name='run-all-diagnoses'),
    path('pipeline-status/', PipelineStatusView.as_view(), name='pipeline-status'),
]
