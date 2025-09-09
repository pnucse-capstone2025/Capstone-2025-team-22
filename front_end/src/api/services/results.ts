import type { RecentResult } from '@/types';
import { BASE_URL, ENDPOINTS } from '../endpoints';

export const getRecentResults = async (): Promise<RecentResult[]> => {
  const response = await fetch(`${BASE_URL}${ENDPOINTS.RECENT_RESULTS}`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const data = await response.json();
  return data.results || [];
}; 