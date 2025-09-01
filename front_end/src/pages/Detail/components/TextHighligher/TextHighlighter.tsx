import { memo, useMemo } from "react";
import { mockDataAnalysis } from "@/data/mock";

interface Props {
  text: string;
  hoveredKeyword: string | null;
}

const TextHighlighter: React.FC<Props> = ({ text, hoveredKeyword }) => {
  const highlightedText = useMemo(() => {
    if (!hoveredKeyword) return text;

    const contributions =
      mockDataAnalysis.keywordAnalysis.find((k) => k.keyword === hoveredKeyword)
        ?.contributions ?? [];

    if (!contributions || contributions.length === 0) {
      return text;
    }

    const sorted = [...contributions].sort((a, b) => a.start - b.start);

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    sorted.forEach(({ start, end, score }, i) => {
      if (lastIndex < start) {
        parts.push(
          <span key={`plain-${i}`}>{text.slice(lastIndex, start)}</span>
        );
      }

      const opacity = Math.min(score / 10, 1) * 0.8; // 1~10 → 0.1~1.0 사이로 변환
      parts.push(
        <span
          key={`hl-${i}`}
          style={{
            backgroundColor: `rgba(30,144,255,${opacity})`,
            borderRadius: "4px",
            padding: "1px 1px",
          }}
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
  }, [text, hoveredKeyword]);

  return <>{highlightedText}</>;
};

export default memo(TextHighlighter);
