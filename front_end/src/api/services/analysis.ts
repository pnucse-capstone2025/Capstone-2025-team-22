import type { AnalysisResult } from '@/types';
import { BASE_URL, ENDPOINTS } from '../endpoints';

export const extractKeywords = async (text: string): Promise<AnalysisResult> => {
  const response = await fetch(`${BASE_URL}${ENDPOINTS.EXTRACT_KEYWORDS}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({
      text: text,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const result = await response.json();
  return result;
};

export const getAnalysisResult = async (analysisId: string): Promise<AnalysisResult> => {
  const response = await fetch(
    `${BASE_URL}${ENDPOINTS.ANALYSIS_RESULT(analysisId)}`
  );
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const data = await response.json();
  return data;
}; 