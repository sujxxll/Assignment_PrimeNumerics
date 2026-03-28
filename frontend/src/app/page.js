'use client';
import { useState, useEffect } from 'react';
import {
  Box, Container, Typography, Grid, Card, CardContent, Chip, Button,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Paper, LinearProgress, IconButton, Dialog, DialogTitle, DialogContent,
  DialogActions, Alert, Snackbar, Tabs, Tab, CircularProgress, Tooltip,
  Accordion, AccordionSummary, AccordionDetails, Divider, Avatar,
} from '@mui/material';
import {
  Dashboard, People, LocalHospital, Psychology, Biotech, Speed,
  PlayArrow, CheckCircle, Warning, ExpandMore, Science, Hub,
  MedicalServices, Medication, Assignment, TrendingUp, AccessTime,
  PersonSearch, HealthAndSafety, AutoAwesome, Memory,
} from '@mui/icons-material';
import { getPatients, getStats, getDiagnoses, getDiagnosis, runDiagnosis, runAllDiagnoses, getPipelineStatus } from '@/services/api';

function StatCard({ icon, title, value, subtitle, color }) {
  return (
    <Card sx={{ height: '100%', transition: 'transform 0.3s, box-shadow 0.3s', '&:hover': { transform: 'translateY(-4px)', boxShadow: `0 8px 32px ${color}33` } }}>
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Avatar sx={{ bgcolor: `${color}22`, color: color, mr: 2, width: 48, height: 48 }}>{icon}</Avatar>
          <Box>
            <Typography variant="body2" color="text.secondary">{title}</Typography>
            <Typography variant="h4" sx={{ color, fontWeight: 800 }}>{value}</Typography>
          </Box>
        </Box>
        {subtitle && <Typography variant="caption" color="text.secondary">{subtitle}</Typography>}
      </CardContent>
    </Card>
  );
}

function PipelineVisualization({ status }) {
  if (!status) return null;
  const steps = [
    { name: 'BioBERT', desc: status?.bert_engine?.model || 'NER Engine', icon: <Biotech />, active: status?.bert_engine?.status === 'active' },
    { name: 'GraphRAG', desc: `${status?.graph_rag?.stats?.nodes || 0} nodes`, icon: <Hub />, active: status?.graph_rag?.status === 'active' },
    { name: 'LLM Ensemble', desc: ' -Med42- ', icon: <Psychology />, active: status?.llm_engine?.status === 'active' },
  ];
  return (
    <Card sx={{ mb: 3, background: 'linear-gradient(135deg, rgba(108,99,255,0.08) 0%, rgba(0,212,170,0.08) 100%)' }}>
      <CardContent sx={{ p: 3 }}>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <AutoAwesome sx={{ color: '#FFB347' }} /> AI Pipeline Status
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-around', flexWrap: 'wrap', gap: 2 }}>
          {steps.map((step, i) => (
            <Box key={step.name} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{
                textAlign: 'center', p: 2, borderRadius: 3,
                border: `2px solid ${step.active ? '#00D4AA' : '#333'}`,
                background: step.active ? 'rgba(0,212,170,0.1)' : 'transparent',
                minWidth: 160, transition: 'all 0.3s',
              }}>
                <Box sx={{ color: step.active ? '#00D4AA' : '#666', mb: 1 }}>{step.icon}</Box>
                <Typography variant="subtitle2" fontWeight={700}>{step.name}</Typography>
                <Typography variant="caption" color="text.secondary">{step.desc}</Typography>
                <Chip size="small" label={step.active ? 'Active' : 'Inactive'}
                  color={step.active ? 'success' : 'default'} sx={{ mt: 1, fontSize: '0.65rem' }} />
              </Box>
              {i < steps.length - 1 && (
                <Typography sx={{ color: '#6C63FF', fontSize: 24, mx: 1 }}>→</Typography>
              )}
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
}

export default function HomePage() {
  const [tab, setTab] = useState(0);
  const [patients, setPatients] = useState([]);
  const [stats, setStats] = useState(null);
  const [diagnoses, setDiagnoses] = useState([]);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [diagnosing, setDiagnosing] = useState(false);
  const [selectedDiagnosis, setSelectedDiagnosis] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [runningPatient, setRunningPatient] = useState(null);

  useEffect(() => { loadData(); }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [pRes, sRes, dRes, psRes] = await Promise.all([
        getPatients().catch(() => ({ data: { results: [] } })),
        getStats().catch(() => ({ data: {} })),
        getDiagnoses().catch(() => ({ data: { results: [] } })),
        getPipelineStatus().catch(() => ({ data: null })),
      ]);
      setPatients(pRes.data.results || pRes.data || []);
      setStats(sRes.data);
      setDiagnoses(dRes.data.results || dRes.data || []);
      setPipelineStatus(psRes.data);
    } catch (e) { console.error('Load error:', e); }
    setLoading(false);
  };

  const handleRunDiagnosis = async (patientId) => {
    setRunningPatient(patientId);
    try {
      const res = await runDiagnosis(patientId);
      setSnackbar({ open: true, message: `Diagnosis completed for ${patientId}`, severity: 'success' });
      loadData();
    } catch (e) {
      setSnackbar({ open: true, message: `Error: ${e.response?.data?.error || e.message}`, severity: 'error' });
    }
    setRunningPatient(null);
  };

  const handleRunAll = async () => {
    setDiagnosing(true);
    try {
      const res = await runAllDiagnoses(true);
      setSnackbar({ open: true, message: `Processed ${res.data.total_processed} patients`, severity: 'success' });
      loadData();
    } catch (e) {
      setSnackbar({ open: true, message: `Error: ${e.message}`, severity: 'error' });
    }
    setDiagnosing(false);
  };

  const handleViewDiagnosis = async (diagId) => {
    try {
      const res = await getDiagnosis(diagId);
      setSelectedDiagnosis(res.data);
      setDialogOpen(true);
    } catch (e) {
      setSnackbar({ open: true, message: 'Failed to load diagnosis details', severity: 'error' });
    }
  };

  if (loading) return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', flexDirection: 'column', gap: 3 }}>
      <CircularProgress size={60} sx={{ color: '#6C63FF' }} />
      <Typography variant="h6" color="text.secondary">Loading HealthAI System...</Typography>
    </Box>
  );

  return (
    <Box sx={{ minHeight: '100vh', background: 'linear-gradient(180deg, #0A0E1A 0%, #0F1629 100%)' }}>
      {/* Header */}
      <Box sx={{ background: 'linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(0,212,170,0.1) 100%)', borderBottom: '1px solid rgba(108,99,255,0.2)', px: 4, py: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <HealthAndSafety sx={{ fontSize: 40, color: '#6C63FF' }} />
            <Box>
              <Typography variant="h5" sx={{ fontWeight: 800, background: 'linear-gradient(135deg, #6C63FF, #00D4AA)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                HealthAI Diagnosis System
              </Typography>
              <Typography variant="caption" color="text.secondary">
                BioBERT → GraphRAG → LLM Ensemble | Powered by  Med42
              </Typography>
            </Box>
          </Box>
          <Button variant="contained" startIcon={diagnosing ? <CircularProgress size={18} color="inherit" /> : <PlayArrow />}
            onClick={handleRunAll} disabled={diagnosing} size="large"
            sx={{ px: 4, py: 1.5, fontSize: '0.95rem' }}>
            {diagnosing ? 'Processing All...' : 'Run All Diagnoses'}
          </Button>
        </Box>
      </Box>

      <Container maxWidth="xl" sx={{ py: 4 }}>
        {/* Stats */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <StatCard icon={<People />} title="Total Patients" value={stats?.total_patients || 0} subtitle="Discharge summaries loaded" color="#6C63FF" />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <StatCard icon={<LocalHospital />} title="Diagnoses Generated" value={stats?.total_diagnoses || 0} subtitle="AI-powered analyses" color="#00D4AA" />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <StatCard icon={<Speed />} title="Avg Confidence" value={`${((stats?.average_confidence || 0) * 100).toFixed(0)}%`} subtitle="Diagnosis precision" color="#FFB347" />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <StatCard icon={<Assignment />} title="Pending" value={stats?.pending_diagnoses || 0} subtitle="Awaiting analysis" color="#FF6B6B" />
          </Grid>
        </Grid>

        {/* Pipeline Status */}
        <PipelineVisualization status={pipelineStatus} />

        {/* Tabs */}
        <Tabs value={tab} onChange={(e, v) => setTab(v)} sx={{ mb: 3, '& .MuiTab-root': { fontWeight: 600, fontSize: '0.95rem' } }}>
          <Tab icon={<People />} label="Patients" iconPosition="start" />
          <Tab icon={<MedicalServices />} label="Diagnoses" iconPosition="start" />
        </Tabs>

        {/* Patients Tab */}
        {tab === 0 && (
          <TableContainer component={Paper} sx={{ background: 'rgba(17,24,39,0.8)', border: '1px solid rgba(108,99,255,0.15)' }}>
            <Table>
              <TableHead>
                <TableRow sx={{ '& th': { color: '#9CA3AF', fontWeight: 700, borderBottom: '1px solid rgba(108,99,255,0.2)' } }}>
                  <TableCell>ID</TableCell><TableCell>Patient Name</TableCell><TableCell>Age</TableCell>
                  <TableCell>Gender</TableCell><TableCell>Department</TableCell><TableCell>Admission</TableCell>
                  <TableCell>Status</TableCell><TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {patients.map((p) => (
                  <TableRow key={p.id} sx={{ '&:hover': { background: 'rgba(108,99,255,0.05)' }, transition: 'background 0.2s' }}>
                    <TableCell><Chip label={p.patient_id} size="small" sx={{ fontWeight: 700, bgcolor: 'rgba(108,99,255,0.15)', color: '#8B83FF' }} /></TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>{p.first_name} {p.last_name}</TableCell>
                    <TableCell>{p.age}</TableCell>
                    <TableCell>{p.gender === 'M' ? '♂ Male' : p.gender === 'F' ? '♀ Female' : 'Other'}</TableCell>
                    <TableCell><Chip label={p.department} size="small" variant="outlined" sx={{ borderColor: '#00D4AA', color: '#00D4AA' }} /></TableCell>
                    <TableCell sx={{ color: '#9CA3AF' }}>{p.admission_date}</TableCell>
                    <TableCell>
                      {p.has_diagnosis
                        ? <Chip icon={<CheckCircle />} label="Diagnosed" size="small" color="success" />
                        : <Chip icon={<Warning />} label="Pending" size="small" sx={{ bgcolor: 'rgba(255,179,71,0.15)', color: '#FFB347' }} />
                      }
                    </TableCell>
                    <TableCell align="center">
                      <Button size="small" variant="outlined" startIcon={
                        runningPatient === p.patient_id ? <CircularProgress size={14} /> : <Science />
                      } onClick={() => handleRunDiagnosis(p.patient_id)}
                        disabled={runningPatient === p.patient_id}
                        sx={{ borderColor: '#6C63FF', color: '#6C63FF', mr: 1 }}>
                        {runningPatient === p.patient_id ? 'Running...' : 'Diagnose'}
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {/* Diagnoses Tab */}
        {tab === 1 && (
          <Grid container spacing={3}>
            {diagnoses.map((d) => (
              <Grid size={{ xs: 12, md: 6, lg: 4 }} key={d.id}>
                <Card sx={{ cursor: 'pointer', transition: 'all 0.3s', '&:hover': { transform: 'translateY(-6px)', boxShadow: '0 12px 40px rgba(108,99,255,0.2)' } }}
                  onClick={() => handleViewDiagnosis(d.id)}>
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Chip label={d.patient_id_str} size="small" sx={{ fontWeight: 700, bgcolor: 'rgba(108,99,255,0.15)', color: '#8B83FF' }} />
                      <Chip label={`${(d.diagnosis_confidence * 100).toFixed(0)}%`} size="small"
                        sx={{
                          fontWeight: 700, bgcolor: d.diagnosis_confidence > 0.8 ? 'rgba(0,212,170,0.15)' : 'rgba(255,179,71,0.15)',
                          color: d.diagnosis_confidence > 0.8 ? '#00D4AA' : '#FFB347'
                        }} />
                    </Box>
                    <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1 }}>{d.patient_name}</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                      {d.primary_diagnosis}
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        {(d.models_used || []).slice(0, 3).map((m) => (
                          <Chip key={m} label={m} size="small" sx={{ fontSize: '0.6rem', height: 20 }} />
                        ))}
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        <AccessTime sx={{ fontSize: 12, mr: 0.5, verticalAlign: 'middle' }} />
                        {d.processing_time_seconds?.toFixed(2)}s
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
            {diagnoses.length === 0 && (
              <Grid size={12}>
                <Card sx={{ textAlign: 'center', py: 6 }}>
                  <MedicalServices sx={{ fontSize: 64, color: '#333', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary">No diagnoses generated yet</Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>Run the AI pipeline to generate diagnoses</Typography>
                  <Button variant="contained" startIcon={<PlayArrow />} onClick={handleRunAll}>Run All Diagnoses</Button>
                </Card>
              </Grid>
            )}
          </Grid>
        )}
      </Container>

      {/* Diagnosis Detail Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="lg" fullWidth
        PaperProps={{ sx: { background: '#111827', border: '1px solid rgba(108,99,255,0.2)', borderRadius: 3, maxHeight: '90vh' } }}>
        {selectedDiagnosis && (
          <>
            <DialogTitle sx={{ background: 'linear-gradient(135deg, rgba(108,99,255,0.1), rgba(0,212,170,0.1))', borderBottom: '1px solid rgba(108,99,255,0.15)' }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography variant="h5" fontWeight={800}>AI Diagnosis Report</Typography>
                  <Typography variant="body2" color="text.secondary">Patient: {selectedDiagnosis.patient_name} ({selectedDiagnosis.patient_id_str})</Typography>
                </Box>
                <Chip label={`${(selectedDiagnosis.diagnosis_confidence * 100).toFixed(0)}% Confidence`}
                  sx={{
                    fontWeight: 700, fontSize: '0.9rem', py: 2,
                    bgcolor: selectedDiagnosis.diagnosis_confidence > 0.8 ? 'rgba(0,212,170,0.2)' : 'rgba(255,179,71,0.2)',
                    color: selectedDiagnosis.diagnosis_confidence > 0.8 ? '#00D4AA' : '#FFB347'
                  }} />
              </Box>
            </DialogTitle>
            <DialogContent sx={{ p: 4 }}>
              {/* Primary Diagnosis */}
              <Accordion defaultExpanded sx={{ mb: 2, background: 'rgba(108,99,255,0.05)', border: '1px solid rgba(108,99,255,0.15)' }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <LocalHospital sx={{ mr: 1, color: '#6C63FF' }} />
                  <Typography fontWeight={700}>Primary Diagnosis</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="h6" sx={{ color: '#00D4AA', mb: 1 }}>{selectedDiagnosis.primary_diagnosis}</Typography>
                  {selectedDiagnosis.secondary_diagnoses?.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>Secondary Diagnoses:</Typography>
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        {selectedDiagnosis.secondary_diagnoses.map((d, i) => (
                          <Chip key={i} label={d} size="small" variant="outlined" sx={{ borderColor: '#FFB347', color: '#FFB347' }} />
                        ))}
                      </Box>
                    </Box>
                  )}
                </AccordionDetails>
              </Accordion>

              {/* Clinical Reasoning */}
              <Accordion defaultExpanded sx={{ mb: 2, background: 'rgba(0,212,170,0.05)', border: '1px solid rgba(0,212,170,0.15)' }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Psychology sx={{ mr: 1, color: '#00D4AA' }} />
                  <Typography fontWeight={700}>Clinical Reasoning</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.8 }}>
                    {selectedDiagnosis.diagnosis_reasoning}
                  </Typography>
                </AccordionDetails>
              </Accordion>

              {/* Treatment Plan */}
              <Accordion defaultExpanded sx={{ mb: 2, background: 'rgba(255,179,71,0.05)', border: '1px solid rgba(255,179,71,0.15)' }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Medication sx={{ mr: 1, color: '#FFB347' }} />
                  <Typography fontWeight={700}>Treatment Plan</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.8, mb: 2 }}>
                    {selectedDiagnosis.treatment_plan}
                  </Typography>
                  {selectedDiagnosis.medications_recommended?.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1 }}>Medications Recommended:</Typography>
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        {selectedDiagnosis.medications_recommended.map((m, i) => (
                          <Chip key={i} label={typeof m === 'string' ? m : m.name} size="small"
                            sx={{ bgcolor: 'rgba(108,99,255,0.1)', color: '#8B83FF' }} />
                        ))}
                      </Box>
                    </Box>
                  )}
                  {selectedDiagnosis.lifestyle_modifications?.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1 }}>Lifestyle Modifications:</Typography>
                      {selectedDiagnosis.lifestyle_modifications.map((m, i) => (
                        <Typography key={i} variant="body2" sx={{ ml: 2 }}>• {m}</Typography>
                      ))}
                    </Box>
                  )}
                </AccordionDetails>
              </Accordion>

              {/* Extracted Entities */}
              <Accordion sx={{ mb: 2, background: 'rgba(108,99,255,0.03)', border: '1px solid rgba(108,99,255,0.1)' }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Biotech sx={{ mr: 1, color: '#6C63FF' }} />
                  <Typography fontWeight={700}>BioBERT Extracted Entities</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {selectedDiagnosis.extracted_entities && Object.entries(selectedDiagnosis.extracted_entities).map(([type, entities]) => (
                    <Box key={type} sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" fontWeight={700} sx={{ textTransform: 'capitalize', mb: 0.5, color: '#9CA3AF' }}>
                        {type.replace('_', ' ')} ({entities.length})
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        {entities.map((e, i) => (
                          <Chip key={i} label={e} size="small" variant="outlined"
                            sx={{ fontSize: '0.7rem', borderColor: type === 'disease' ? '#FF6B6B' : type === 'medication' ? '#6C63FF' : type === 'symptom' ? '#FFB347' : '#00D4AA' }} />
                        ))}
                      </Box>
                    </Box>
                  ))}
                </AccordionDetails>
              </Accordion>

              {/* Models Used */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 3, pt: 2, borderTop: '1px solid rgba(108,99,255,0.15)' }}>
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  <Memory sx={{ color: '#9CA3AF', fontSize: 18 }} />
                  <Typography variant="caption" color="text.secondary">Models: </Typography>
                  {(selectedDiagnosis.models_used || []).map((m) => (
                    <Chip key={m} label={m} size="small" sx={{ fontSize: '0.65rem', bgcolor: 'rgba(108,99,255,0.1)' }} />
                  ))}
                </Box>
                <Typography variant="caption" color="text.secondary">
                  Processing time: {selectedDiagnosis.processing_time_seconds?.toFixed(3)}s
                </Typography>
              </Box>
            </DialogContent>
            <DialogActions sx={{ px: 4, py: 2, borderTop: '1px solid rgba(108,99,255,0.15)' }}>
              <Button onClick={() => setDialogOpen(false)} variant="outlined">Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      <Snackbar open={snackbar.open} autoHideDuration={5000} onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}>
        <Alert severity={snackbar.severity} onClose={() => setSnackbar({ ...snackbar, open: false })} sx={{ borderRadius: 2 }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
