import uuid
from django.db import models


class Patient(models.Model):
    """Represents a patient in the healthcare system."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient_id = models.CharField(max_length=20, unique=True, db_index=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10, choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')])
    admission_date = models.DateField()
    discharge_date = models.DateField()
    department = models.CharField(max_length=100)
    attending_physician = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-admission_date']

    def __str__(self):
        return f"{self.patient_id} - {self.first_name} {self.last_name}"


class DischargeSummary(models.Model):
    """Stores discharge summary text for each patient."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.OneToOneField(Patient, on_delete=models.CASCADE, related_name='discharge_summary')
    chief_complaint = models.TextField()
    history_of_present_illness = models.TextField()
    past_medical_history = models.TextField(blank=True, default='')
    medications_on_admission = models.TextField(blank=True, default='')
    physical_examination = models.TextField(blank=True, default='')
    lab_results = models.TextField(blank=True, default='')
    imaging_results = models.TextField(blank=True, default='')
    hospital_course = models.TextField()
    discharge_diagnosis = models.TextField(blank=True, default='')
    discharge_medications = models.TextField(blank=True, default='')
    follow_up_instructions = models.TextField(blank=True, default='')
    full_summary_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = 'Discharge summaries'

    def __str__(self):
        return f"Discharge Summary - {self.patient.patient_id}"


class Diagnosis(models.Model):
    """AI-generated diagnosis for a patient."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='diagnoses')
    discharge_summary = models.ForeignKey(DischargeSummary, on_delete=models.CASCADE, related_name='diagnoses')

    # AI Pipeline Results
    extracted_entities = models.JSONField(default=dict, help_text='BioBERT extracted entities')
    graph_context = models.JSONField(default=dict, help_text='GraphRAG retrieved context')
    
    # LLM Results
    primary_diagnosis = models.TextField()
    secondary_diagnoses = models.JSONField(default=list)
    diagnosis_confidence = models.FloatField(default=0.0)
    diagnosis_reasoning = models.TextField()
    
    # Treatment Plan
    treatment_plan = models.TextField()
    medications_recommended = models.JSONField(default=list)
    procedures_recommended = models.JSONField(default=list)
    lifestyle_modifications = models.JSONField(default=list)
    follow_up_schedule = models.JSONField(default=list)
    
    # Metadata
    models_used = models.JSONField(default=list, help_text='List of AI models used')
    processing_time_seconds = models.FloatField(default=0.0)
    pipeline_version = models.CharField(max_length=20, default='1.0')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Diagnoses'

    def __str__(self):
        return f"Diagnosis for {self.patient.patient_id} - {self.primary_diagnosis[:50]}"


class KnowledgeGraphNode(models.Model):
    """Stores nodes in the medical knowledge graph."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    node_type = models.CharField(max_length=50, choices=[
        ('disease', 'Disease'),
        ('symptom', 'Symptom'),
        ('medication', 'Medication'),
        ('procedure', 'Procedure'),
        ('lab_test', 'Lab Test'),
        ('anatomy', 'Anatomy'),
        ('gene', 'Gene'),
    ], db_index=True)
    name = models.CharField(max_length=300, db_index=True)
    description = models.TextField(blank=True, default='')
    embedding = models.JSONField(null=True, blank=True, help_text='Vector embedding for similarity search')
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['node_type', 'name']

    def __str__(self):
        return f"{self.node_type}: {self.name}"


class KnowledgeGraphEdge(models.Model):
    """Stores edges/relationships in the medical knowledge graph."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source = models.ForeignKey(KnowledgeGraphNode, on_delete=models.CASCADE, related_name='outgoing_edges')
    target = models.ForeignKey(KnowledgeGraphNode, on_delete=models.CASCADE, related_name='incoming_edges')
    relationship_type = models.CharField(max_length=100, choices=[
        ('causes', 'Causes'),
        ('treats', 'Treats'),
        ('symptom_of', 'Symptom Of'),
        ('diagnosed_by', 'Diagnosed By'),
        ('contraindicated', 'Contraindicated'),
        ('interacts_with', 'Interacts With'),
        ('associated_with', 'Associated With'),
        ('preceded_by', 'Preceded By'),
    ], db_index=True)
    weight = models.FloatField(default=1.0)
    metadata = models.JSONField(default=dict)

    class Meta:
        unique_together = ['source', 'target', 'relationship_type']

    def __str__(self):
        return f"{self.source.name} --{self.relationship_type}--> {self.target.name}"
