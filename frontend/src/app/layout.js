'use client';
import { ThemeProvider, CssBaseline } from '@mui/material';
import theme from '@/theme/theme';

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <title>HealthAI - Intelligent Clinical Diagnosis System</title>
        <meta name="description" content="AI-powered healthcare diagnosis system using BioBERT, GraphRAG, and LLM ensemble" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body style={{ margin: 0, padding: 0, background: '#0A0E1A' }}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
