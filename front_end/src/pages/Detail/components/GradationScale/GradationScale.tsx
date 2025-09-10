import { useMemo } from "react";
import styles from "./GradationScale.module.scss";

import type {
  GradationScaleProps,
  AttentionResult,
  AttentionItem,
} from "../../../../types";

const GradationScale: React.FC<GradationScaleProps> = ({
  maxScores = { noun: 1.0, verb: 1.0 },
  selectedKeyword,
  clickedWordScore,
  attentionResult,
}) => {
  const normalizedMaxScores = useMemo(() => {
    if (
      !attentionResult ||
      !selectedKeyword ||
      !attentionResult[selectedKeyword]
    ) {
      return { noun: 1.0, verb: 1.0 };
    }

    const keywordData = attentionResult[selectedKeyword];

    // 명사와 동사의 최대 점수 찾기
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
  }, [attentionResult, selectedKeyword]);

  const nounTicks = useMemo(() => {
    const tickCount = 6;
    const ticks = [];
    for (let i = 0; i < tickCount; i++) {
      const value = (maxScores.noun / tickCount) * i;
      ticks.push(value.toFixed(1));
    }
    return ticks;
  }, [maxScores.noun]);

  const verbTicks = useMemo(() => {
    const tickCount = 6;
    const ticks = [];
    for (let i = 0; i < tickCount; i++) {
      const value = (maxScores.verb / tickCount) * i;
      ticks.push(value.toFixed(1));
    }
    return ticks;
  }, [maxScores.verb]);

  const getMarkerPosition = (score: number, maxScore: number) => {
    const position = (score / maxScore) * 100;
    return Math.max(0, Math.min(100, position));
  };

  const getNormalizedHighlightColor = (
    score: number,
    type: "noun" | "verb"
  ) => {
    const maxScore =
      type === "noun" ? normalizedMaxScores.noun : normalizedMaxScores.verb;
    const normalizedScore = score / maxScore;
    const opacity = normalizedScore * 0.8;

    return type === "noun"
      ? `rgba(220, 53, 69, ${opacity})`
      : `rgba(30, 144, 255, ${opacity})`;
  };

  const generateScaleGradient = (maxScore: number, type: "noun" | "verb") => {
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

  return (
    <div className={styles.gradationScale}>
      <div className={styles.title}>
        {selectedKeyword
          ? `"${selectedKeyword}" 키워드 점수 범례`
          : "점수 범례"}
      </div>

      <div className={styles.scaleContainer}>
        {/* 명사 스케일 */}
        <div className={styles.scaleSection}>
          <div className={styles.scaleRow}>
            <div className={styles.label}>
              <span className={`${styles.colorDot} ${styles.nounDot}`}></span>
              명사
            </div>
            <div
              className={styles.scaleBar}
              style={{
                background: generateScaleGradient(maxScores.noun, "noun"),
              }}
            >
              {clickedWordScore?.type === "noun" && (
                <div
                  className={styles.scoreMarker}
                  style={{
                    left: `${getMarkerPosition(
                      clickedWordScore.score,
                      maxScores.noun
                    )}%`,
                    backgroundColor: getNormalizedHighlightColor(
                      clickedWordScore.score,
                      "noun"
                    ),
                    border: "2px solid #ffffff",
                    boxShadow: "0 2px 4px rgba(0, 0, 0, 0.2)",
                  }}
                />
              )}
            </div>
          </div>
          <div className={styles.tickMarks}>
            {nounTicks.map((tick, index) => (
              <span
                key={`noun-${index}`}
                className={styles.tick}
                style={{
                  left: `${(index / (nounTicks.length - 1)) * 100}%`,
                }}
              >
                {tick}
              </span>
            ))}
          </div>
        </div>

        {/* 동사 스케일 */}
        <div className={styles.scaleSection}>
          <div className={styles.scaleRow}>
            <div className={styles.label}>
              <span className={`${styles.colorDot} ${styles.verbDot}`}></span>
              동사
            </div>
            <div
              className={styles.scaleBar}
              style={{
                background: generateScaleGradient(maxScores.verb, "verb"),
              }}
            >
              {clickedWordScore?.type === "verb" && (
                <div
                  className={styles.scoreMarker}
                  style={{
                    left: `${getMarkerPosition(
                      clickedWordScore.score,
                      maxScores.verb
                    )}%`,
                    backgroundColor: getNormalizedHighlightColor(
                      clickedWordScore.score,
                      "verb"
                    ),
                    border: "2px solid #ffffff",
                    boxShadow: "0 2px 4px rgba(0, 0, 0, 0.2)",
                  }}
                />
              )}
            </div>
          </div>
          <div className={styles.tickMarks}>
            {verbTicks.map((tick, index) => (
              <span
                key={`verb-${index}`}
                className={styles.tick}
                style={{
                  left: `${(index / (verbTicks.length - 1)) * 100}%`,
                }}
              >
                {tick}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className={styles.description}>
        점수가 높을수록 키워드 추출에 더 많이 기여한 단어입니다.
      </div>
    </div>
  );
};

export default GradationScale;
