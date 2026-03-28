from django.contrib import admin
from .models import Patient, DischargeSummary, Diagnosis, KnowledgeGraphNode, KnowledgeGraphEdge

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['patient_id', 'first_name', 'last_name', 'age', 'gender', 'department']
    search_fields = ['patient_id', 'first_name', 'last_name']
    list_filter = ['gender', 'department']

@admin.register(DischargeSummary)
class DischargeSummaryAdmin(admin.ModelAdmin):
    list_display = ['patient', 'chief_complaint', 'created_at']
    search_fields = ['patient__patient_id', 'chief_complaint']

@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    list_display = ['patient', 'primary_diagnosis', 'diagnosis_confidence', 'created_at']
    search_fields = ['patient__patient_id', 'primary_diagnosis']
    list_filter = ['pipeline_version']

@admin.register(KnowledgeGraphNode)
class KnowledgeGraphNodeAdmin(admin.ModelAdmin):
    list_display = ['name', 'node_type', 'created_at']
    list_filter = ['node_type']
    search_fields = ['name']

@admin.register(KnowledgeGraphEdge)
class KnowledgeGraphEdgeAdmin(admin.ModelAdmin):
    list_display = ['source', 'relationship_type', 'target', 'weight']
    list_filter = ['relationship_type']
