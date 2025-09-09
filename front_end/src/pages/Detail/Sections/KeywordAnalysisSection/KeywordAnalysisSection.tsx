import { useMemo } from "react";
import styles from "./KeywordAnalysisSection.module.scss";
import TextHighlighter from "@/pages/Detail/components/TextHighligher/TextHighlighter";
import GradationScale from "@/pages/Detail/components/GradationScale/GradationScale";

import type { KeywordAnalysisSectionProps } from "../../../../types";

const KeywordAnalysisSection: React.FC<KeywordAnalysisSectionProps> = ({
  analysisData,
  selectedKeyword,
  clickedWordScore,
  onKeywordSelect,
  onWordClick,
}) => {
  const keywords = useMemo(() => {
    if (!analysisData?.keywords) return [];

    const processedKeywords = analysisData.keywords.map((kw: any) => {
      if (typeof kw === "string") return kw;
      if (typeof kw === "object" && kw && "keyword" in kw) return kw.keyword;
      return String(kw);
    });

    return [...new Set(processedKeywords)];
  }, [analysisData?.keywords]);

  const maxScores = useMemo(() => {
    if (!analysisData?.attention_result || !selectedKeyword) {
      return { noun: 1.0, verb: 1.0 };
    }

    const keywordData = analysisData.attention_result[selectedKeyword];
    let nounMax = 0;
    let verbMax = 0;

    Object.values(keywordData?.nouns || {}).forEach((item) => {
      nounMax = Math.max(nounMax, item.score);
    });
    Object.values(keywordData?.verbs || {}).forEach((item) => {
      verbMax = Math.max(verbMax, item.score);
    });

    return {
      noun: nounMax > 0 ? nounMax : 1.0,
      verb: verbMax > 0 ? verbMax : 1.0,
    };
  }, [analysisData?.attention_result, selectedKeyword]);

  return (
    <div className={styles.keywordAnalysisSection}>
      {keywords.length > 0 ? (
        <div className={styles.keywordButtons}>
          {keywords.map((kw: string, i: number) => (
            <button
              key={i}
              className={`${styles.keywordButton} ${
                selectedKeyword === kw ? styles.selected : ""
              }`}
              onClick={() =>
                onKeywordSelect(selectedKeyword === kw ? null : kw)
              }
            >
              {kw}
            </button>
          ))}
        </div>
      ) : (
        <span className={styles.highlight}>추출된 키워드가 없습니다</span>
      )}

      <div className={styles.textContent}>
        <TextHighlighter
          text={analysisData?.text || ""}
          hoveredKeyword={selectedKeyword}
          attentionResult={analysisData?.attention_result}
          onWordClick={onWordClick}
        />
      </div>

      <div className={styles.gradationScaleWrapper}>
        <GradationScale
          maxScores={maxScores}
          selectedKeyword={selectedKeyword}
          clickedWordScore={clickedWordScore}
        />
      </div>
    </div>
  );
};

export default KeywordAnalysisSection;
