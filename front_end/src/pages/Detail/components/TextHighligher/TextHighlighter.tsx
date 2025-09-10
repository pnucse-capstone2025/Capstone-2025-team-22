import { memo, useMemo } from "react";
import type { AttentionResult, AttentionItem } from "../../../../types";

interface Props {
  text: string;
  hoveredKeyword: string | null;
  attentionResult?: AttentionResult;
  onWordClick?: (score: number, type: "noun" | "verb") => void;
}

const TextHighlighter: React.FC<Props> = ({
  text,
  hoveredKeyword,
  attentionResult,
  onWordClick,
}) => {
  // 품사별 최대 점수 계산 (정규화를 위한)
  const normalizedMaxScores = useMemo(() => {
    if (!attentionResult || !hoveredKeyword || !attentionResult[hoveredKeyword]) {
      return { noun: 1.0, verb: 1.0 };
    }

    const keywordData = attentionResult[hoveredKeyword];
    
    // 명사와 동사의 최대 점수 찾기
    const nounScores = Object.values(keywordData.nouns || {}).map((item: AttentionItem) => item.score);
    const verbScores = Object.values(keywordData.verbs || {}).map((item: AttentionItem) => item.score);
    
    const maxNounScore = nounScores.length > 0 ? Math.max(...nounScores) : 1.0;
    const maxVerbScore = verbScores.length > 0 ? Math.max(...verbScores) : 1.0;
    
    return {
      noun: maxNounScore,
      verb: maxVerbScore
    };
  }, [attentionResult, hoveredKeyword]);

  const highlightedText = useMemo(() => {
    if (
      !hoveredKeyword ||
      !attentionResult ||
      !attentionResult[hoveredKeyword]
    ) {
      return text;
    }

    const keywordData = attentionResult[hoveredKeyword];

    const contributions: (AttentionItem & { type: "noun" | "verb" })[] = [];

    Object.values(keywordData.nouns || {}).forEach((item) =>
      contributions.push({ ...item, type: "noun" })
    );

    Object.values(keywordData.verbs || {}).forEach((item) =>
      contributions.push({ ...item, type: "verb" })
    );

    if (contributions.length === 0) {
      return text;
    }

    const sorted = [...contributions].sort((a, b) => a.start - b.start);

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    sorted.forEach(({ start, end, score, type }, i) => {
      if (lastIndex < start) {
        parts.push(
          <span key={`plain-${i}`}>{text.slice(lastIndex, start)}</span>
        );
      }

      // 정규화된 점수로 투명도 계산
      const maxScore = type === "noun" ? normalizedMaxScores.noun : normalizedMaxScores.verb;
      const normalizedScore = score / maxScore; // 0~1 사이 값으로 정규화
      const opacity = normalizedScore * 0.8; // 최대 80% 투명도

      const backgroundColor =
        type === "noun"
          ? `rgba(220, 53, 69, ${opacity})`
          : `rgba(30, 144, 255, ${opacity})`;

      parts.push(
        <span
          key={`hl-${i}`}
          style={{
            backgroundColor,
            borderRadius: "4px",
            padding: "1px 1px",
            cursor: onWordClick ? "pointer" : "default",
          }}
          onClick={() => onWordClick?.(score, type)}
        >
          {text.slice(start, end)}
        </span>
      );

      lastIndex = end;
    });

    if (lastIndex < text.length) {
      parts.push(<span key="last">{text.slice(lastIndex)}</span>);
    }

    return parts;
  }, [text, hoveredKeyword, attentionResult, onWordClick, normalizedMaxScores]);

  return <>{highlightedText}</>;
};

export default memo(TextHighlighter);
