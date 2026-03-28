from rest_framework import serializers
from .models import Patient, DischargeSummary, Diagnosis, KnowledgeGraphNode, KnowledgeGraphEdge


class DischargeSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = DischargeSummary
        fields = '__all__'


class DischargeSummaryCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = DischargeSummary
        exclude = ['patient']


class PatientListSerializer(serializers.ModelSerializer):
    has_discharge_summary = serializers.SerializerMethodField()
    has_diagnosis = serializers.SerializerMethodField()

    class Meta:
        model = Patient
        fields = [
            'id', 'patient_id', 'first_name', 'last_name', 'age', 'gender',
            'admission_date', 'discharge_date', 'department', 'attending_physician',
            'has_discharge_summary', 'has_diagnosis'
        ]

    def get_has_discharge_summary(self, obj):
        return hasattr(obj, 'discharge_summary')

    def get_has_diagnosis(self, obj):
        return obj.diagnoses.exists()


class PatientDetailSerializer(serializers.ModelSerializer):
    discharge_summary = DischargeSummarySerializer(read_only=True)
    latest_diagnosis = serializers.SerializerMethodField()

    class Meta:
        model = Patient
        fields = '__all__'

    def get_latest_diagnosis(self, obj):
        diagnosis = obj.diagnoses.first()
        if diagnosis:
            return DiagnosisSerializer(diagnosis).data
        return None


class DiagnosisSerializer(serializers.ModelSerializer):
    patient_name = serializers.SerializerMethodField()
    patient_id_str = serializers.CharField(source='patient.patient_id', read_only=True)

    class Meta:
        model = Diagnosis
        fields = '__all__'

    def get_patient_name(self, obj):
        return f"{obj.patient.first_name} {obj.patient.last_name}"


class DiagnosisListSerializer(serializers.ModelSerializer):
    patient_name = serializers.SerializerMethodField()
    patient_id_str = serializers.CharField(source='patient.patient_id', read_only=True)

    class Meta:
        model = Diagnosis
        fields = [
            'id', 'patient', 'patient_name', 'patient_id_str',
            'primary_diagnosis', 'diagnosis_confidence',
            'models_used', 'processing_time_seconds', 'created_at'
        ]

    def get_patient_name(self, obj):
        return f"{obj.patient.first_name} {obj.patient.last_name}"


class KnowledgeGraphNodeSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeGraphNode
        exclude = ['embedding']


class KnowledgeGraphEdgeSerializer(serializers.ModelSerializer):
    source_name = serializers.CharField(source='source.name', read_only=True)
    target_name = serializers.CharField(source='target.name', read_only=True)
    source_type = serializers.CharField(source='source.node_type', read_only=True)
    target_type = serializers.CharField(source='target.node_type', read_only=True)

    class Meta:
        model = KnowledgeGraphEdge
        fields = '__all__'
