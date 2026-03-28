import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000/api';

const api = axios.create({ baseURL: API_BASE });

export const getPatients = () => api.get('/patients/');
export const getPatient = (id) => api.get(`/patients/${id}/`);
export const getStats = () => api.get('/patients/stats/');

export const getDiagnoses = () => api.get('/diagnoses/');
export const getDiagnosis = (id) => api.get(`/diagnoses/${id}/`);

export const runDiagnosis = (patientId) => api.post(`/diagnose/${patientId}/`);
export const runAllDiagnoses = (force = false) => api.post('/diagnose-all/', { force });

export const getPipelineStatus = () => api.get('/pipeline-status/');

export const getKnowledgeGraph = () => api.get('/knowledge-graph/graph_data/');
export const getGraphStats = () => api.get('/knowledge-graph/stats/');

export default api;
