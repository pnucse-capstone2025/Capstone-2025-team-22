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
}

const TextHighlighter: React.FC<Props> = ({
  text,
  hoveredKeyword,
  attentionResult,
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
    const contributions: AttentionItem[] = [];

    // nouns와 verbs에서 모든 attention 아이템들을 수집
    Object.values(keywordData.nouns || {}).forEach((item) =>
      contributions.push(item)
    );
    Object.values(keywordData.verbs || {}).forEach((item) =>
      contributions.push(item)
    );

    if (contributions.length === 0) {
      return text;
    }

    // start 위치로 정렬
    const sorted = [...contributions].sort((a, b) => a.start - b.start);

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    sorted.forEach(({ start, end, score }, i) => {
      if (lastIndex < start) {
        parts.push(
          <span key={`plain-${i}`}>{text.slice(lastIndex, start)}</span>
        );
      }

      // score를 0-1 범위로 정규화 (최대 score가 1이라고 가정)
      const opacity = Math.min(score, 1) * 0.8;
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
  }, [text, hoveredKeyword, attentionResult]);

  return <>{highlightedText}</>;
};

export default memo(TextHighlighter);
