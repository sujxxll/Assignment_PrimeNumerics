import json
import os
from django.core.management.base import BaseCommand
from patients.models import Patient, DischargeSummary


class Command(BaseCommand):
    help = 'Seed database with 30 patient discharge summaries'

    def handle(self, *args, **options):
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
        files = ['patients_1_10.json', 'patients_11_20.json', 'patients_21_30.json']

        count = 0
        for fname in files:
            fpath = os.path.join(data_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                patients_data = json.load(f)

            for p in patients_data:
                patient, created = Patient.objects.update_or_create(
                    patient_id=p['patient_id'],
                    defaults={
                        'first_name': p['first_name'],
                        'last_name': p['last_name'],
                        'age': p['age'],
                        'gender': p['gender'],
                        'admission_date': p['admission_date'],
                        'discharge_date': p['discharge_date'],
                        'department': p['department'],
                        'attending_physician': p['attending_physician'],
                    }
                )

                s = p['summary']
                full_text = self._build_full_text(p, s)

                DischargeSummary.objects.update_or_create(
                    patient=patient,
                    defaults={
                        'chief_complaint': s['chief_complaint'],
                        'history_of_present_illness': s['history_of_present_illness'],
                        'past_medical_history': s.get('past_medical_history', ''),
                        'medications_on_admission': s.get('medications_on_admission', ''),
                        'physical_examination': s.get('physical_examination', ''),
                        'lab_results': s.get('lab_results', ''),
                        'imaging_results': s.get('imaging_results', ''),
                        'hospital_course': s['hospital_course'],
                        'discharge_diagnosis': s.get('discharge_diagnosis', ''),
                        'discharge_medications': s.get('discharge_medications', ''),
                        'follow_up_instructions': s.get('follow_up_instructions', ''),
                        'full_summary_text': full_text,
                    }
                )
                count += 1
                action = 'Created' if created else 'Updated'
                self.stdout.write(f"  {action}: {p['patient_id']} - {p['first_name']} {p['last_name']}")

        self.stdout.write(self.style.SUCCESS(f'\nSuccessfully seeded {count} patients'))

    def _build_full_text(self, p, s):
        sections = [
            f"Patient: {p['first_name']} {p['last_name']}, Age: {p['age']}, Gender: {p['gender']}",
            f"Department: {p['department']}, Physician: {p['attending_physician']}",
            f"Admission: {p['admission_date']} - Discharge: {p['discharge_date']}",
            f"\nChief Complaint: {s['chief_complaint']}",
            f"\nHistory of Present Illness: {s['history_of_present_illness']}",
            f"\nPast Medical History: {s.get('past_medical_history', 'N/A')}",
            f"\nMedications on Admission: {s.get('medications_on_admission', 'N/A')}",
            f"\nPhysical Examination: {s.get('physical_examination', 'N/A')}",
            f"\nLab Results: {s.get('lab_results', 'N/A')}",
            f"\nImaging Results: {s.get('imaging_results', 'N/A')}",
            f"\nHospital Course: {s['hospital_course']}",
            f"\nDischarge Diagnosis: {s.get('discharge_diagnosis', 'N/A')}",
            f"\nDischarge Medications: {s.get('discharge_medications', 'N/A')}",
            f"\nFollow-up Instructions: {s.get('follow_up_instructions', 'N/A')}",
        ]
        return '\n'.join(sections)
