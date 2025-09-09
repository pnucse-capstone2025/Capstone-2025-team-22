export const BASE_URL = 'http://localhost:8000';

export const ENDPOINTS = {
  RECENT_RESULTS: '/recent_results',
  EXTRACT_KEYWORDS: '/extract_keywords',
  ANALYSIS_RESULT: (id: string) => `/analysis_results/${id}`,
} as const; 