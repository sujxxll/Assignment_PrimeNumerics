'use client';
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#6C63FF', light: '#8B83FF', dark: '#4A42D4' },
    secondary: { main: '#00D4AA', light: '#33DDBB', dark: '#00A888' },
    background: { default: '#0A0E1A', paper: '#111827' },
    error: { main: '#FF6B6B' },
    warning: { main: '#FFB347' },
    success: { main: '#00D4AA' },
    info: { main: '#6C63FF' },
    text: { primary: '#E8EAED', secondary: '#9CA3AF' },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", sans-serif',
    h4: { fontWeight: 700 },
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
  },
  shape: { borderRadius: 16 },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(135deg, rgba(108,99,255,0.05) 0%, rgba(0,212,170,0.05) 100%)',
          border: '1px solid rgba(108,99,255,0.15)',
          backdropFilter: 'blur(10px)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: { textTransform: 'none', fontWeight: 600, borderRadius: 12 },
        containedPrimary: {
          background: 'linear-gradient(135deg, #6C63FF 0%, #8B83FF 100%)',
          '&:hover': { background: 'linear-gradient(135deg, #5A52E0 0%, #7A73FF 100%)' },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { borderRadius: 8 },
      },
    },
  },
});

export default theme;
