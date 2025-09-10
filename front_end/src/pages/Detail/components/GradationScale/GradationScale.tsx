import { useMemo } from "react";
import styles from "@/pages/Detail/components/GradationScale/GradationScale.module.scss";
import {
  calculateNormalizedMaxScores,
  generateTicks,
  getMarkerPosition,
  getNormalizedHighlightColor,
  generateScaleGradient,
} from "@/utils/gradationScaleUtils";

import type { GradationScaleProps } from "@/types";

const GradationScale: React.FC<GradationScaleProps> = ({
  maxScores = { noun: 1.0, verb: 1.0 },
  selectedKeyword,
  clickedWordScore,
  attentionResult,
}) => {
  const normalizedMaxScores = useMemo(
    () =>
      calculateNormalizedMaxScores(attentionResult, selectedKeyword || null),
    [attentionResult, selectedKeyword]
  );

  const nounTicks = useMemo(
    () => generateTicks(maxScores.noun),
    [maxScores.noun]
  );
  const verbTicks = useMemo(
    () => generateTicks(maxScores.verb),
    [maxScores.verb]
  );

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
                background: generateScaleGradient("noun"),
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
                      "noun",
                      normalizedMaxScores
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
                background: generateScaleGradient("verb"),
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
                      "verb",
                      normalizedMaxScores
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
