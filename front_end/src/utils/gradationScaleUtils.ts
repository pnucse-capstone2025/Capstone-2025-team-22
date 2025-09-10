import type { AttentionResult, AttentionItem } from "@/types";

export const calculateNormalizedMaxScores = (
  attentionResult: AttentionResult | undefined,
  selectedKeyword: string | null
) => {
  if (
    !attentionResult ||
    !selectedKeyword ||
    !attentionResult[selectedKeyword]
  ) {
    return { noun: 1.0, verb: 1.0 };
  }

  const keywordData = attentionResult[selectedKeyword];

  const nounScores = Object.values(keywordData.nouns || {}).map(
    (item: AttentionItem) => item.score
  );
  const verbScores = Object.values(keywordData.verbs || {}).map(
    (item: AttentionItem) => item.score
  );

  const maxNounScore = nounScores.length > 0 ? Math.max(...nounScores) : 1.0;
  const maxVerbScore = verbScores.length > 0 ? Math.max(...verbScores) : 1.0;

  return {
    noun: maxNounScore,
    verb: maxVerbScore,
  };
};

export const generateTicks = (maxScore: number, tickCount: number = 6) => {
  const ticks = [];
  for (let i = 0; i < tickCount; i++) {
    const value = (maxScore / tickCount) * i;
    ticks.push(value.toFixed(1));
  }
  return ticks;
};

export const getMarkerPosition = (score: number, maxScore: number) => {
  const position = (score / maxScore) * 100;
  return Math.max(0, Math.min(100, position));
};

export const getNormalizedHighlightColor = (
  score: number,
  type: "noun" | "verb",
  normalizedMaxScores: { noun: number; verb: number }
) => {
  const maxScore =
    type === "noun" ? normalizedMaxScores.noun : normalizedMaxScores.verb;
  const normalizedScore = score / maxScore;
  const opacity = normalizedScore * 0.8;

  return type === "noun"
    ? `rgba(220, 53, 69, ${opacity})`
    : `rgba(30, 144, 255, ${opacity})`;
};

export const generateScaleGradient = (
  maxScore: number,
  type: "noun" | "verb",
  normalizedMaxScores: { noun: number; verb: number }
) => {
  const baseColor = type === "noun" ? "220, 53, 69" : "30, 144, 255";
  const normalizedMaxScore =
    type === "noun" ? normalizedMaxScores.noun : normalizedMaxScores.verb;
  const steps = [];

  for (let i = 0; i <= 4; i++) {
    const ratio = i / 4;
    const score = maxScore * ratio;
    const normalizedScore = score / normalizedMaxScore;
    const opacity = Math.min(normalizedScore, 1) * 0.8;
    steps.push(`rgba(${baseColor}, ${opacity}) ${ratio * 100}%`);
  }

  return `linear-gradient(to right, ${steps.join(", ")})`;
};
