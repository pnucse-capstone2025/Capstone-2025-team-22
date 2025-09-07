import { memo, useMemo } from "react";

interface AttentionResult {
  [keyword: string]: {
    nouns: { [key: string]: AttentionItem };
    verbs: { [key: string]: AttentionItem };
  };
}

interface AttentionItem {
  keyword: string;
  score: number;
  start: number;
  end: number;
}

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

      const opacity = Math.min(score, 1) * 0.8;

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
  }, [text, hoveredKeyword, attentionResult, onWordClick]);

  return <>{highlightedText}</>;
};

export default memo(TextHighlighter);
