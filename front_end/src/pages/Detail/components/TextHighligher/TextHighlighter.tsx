import { memo, useMemo } from "react";
import type { AttentionResult, AttentionItem } from "@/types";
import {
  calculateNormalizedMaxScores,
  getNormalizedHighlightColor,
} from "@/utils/gradationScaleUtils";

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
  const normalizedMaxScores = useMemo(
    () => calculateNormalizedMaxScores(attentionResult, hoveredKeyword),
    [attentionResult, hoveredKeyword]
  );

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
      const backgroundColor = getNormalizedHighlightColor(
        score,
        type,
        normalizedMaxScores
      );

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
