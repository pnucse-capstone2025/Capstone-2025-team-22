export interface AnalysisResult {
  text: string;
  nouns: string[];
  verbs: string[];
  adjectives: string[];
  keywords: string[];
  noun_count: number;
  verb_count: number;
  adjective_count: number;
  attention_result?: AttentionResult;
}

export interface RecentResult {
  id: number;
  text: string;
  created_at: string;
  nouns: string[];
  verbs: string[];
  adjectives: string[];
  keywords: string[];
}

export interface AttentionResult {
  [keyword: string]: {
    nouns: { [key: string]: AttentionItem };
    verbs: { [key: string]: AttentionItem };
  };
}

export interface AttentionItem {
  keyword: string;
  score: number;
  start: number;
  end: number;
}
