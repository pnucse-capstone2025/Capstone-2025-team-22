import type { AttentionResult, RecentResult } from "./api";
import type { ClickedWordScore } from "./analysis";

export interface DrawerProps {
  isOpen: boolean;
  onToggle: () => void;
  recentResults: RecentResult[];
}

export interface KeywordAnalysisSectionProps {
  analysisData: {
    text: string;
    keywords: string[];
    attention_result?: AttentionResult;
  } | null;
  selectedKeyword: string | null;
  clickedWordScore: ClickedWordScore | null;
  onKeywordSelect: (keyword: string | null) => void;
  onWordClick: (score: number, type: "noun" | "verb") => void;
}

export interface GradationScaleProps {
  maxScores?: { noun: number; verb: number };
  selectedKeyword?: string | null;
  clickedWordScore?: ClickedWordScore | null;
  attentionResult?: AttentionResult;
}
