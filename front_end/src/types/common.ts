export type WordType = "noun" | "verb" | "adjective" | "adverb";

export type AnalysisStatus = "pending" | "processing" | "completed" | "error";

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: any;
}
