import { useMemo } from "react";
import styles from "./GradationScale.module.scss";

interface GradationScaleProps {
  maxScores?: { noun: number; verb: number };
  selectedKeyword?: string | null;
  clickedWordScore?: { score: number; type: "noun" | "verb" } | null;
}

const GradationScale: React.FC<GradationScaleProps> = ({
  maxScores = { noun: 1.0, verb: 1.0 },
  selectedKeyword,
  clickedWordScore,
}) => {
  // 명사와 동사별 스케일 눈금 생성
  const nounTicks = useMemo(() => {
    const tickCount = 6;
    const ticks = [];
    for (let i = 0; i < tickCount; i++) {
      const value = (maxScores.noun / (tickCount - 1)) * i;
      ticks.push(value.toFixed(1));
    }
    return ticks;
  }, [maxScores.noun]);

  const verbTicks = useMemo(() => {
    const tickCount = 6;
    const ticks = [];
    for (let i = 0; i < tickCount; i++) {
      const value = (maxScores.verb / (tickCount - 1)) * i;
      ticks.push(value.toFixed(1));
    }
    return ticks;
  }, [maxScores.verb]);

  const getMarkerPosition = (score: number, maxScore: number) => {
    const position = (score / maxScore) * 100;
    return Math.max(0, Math.min(100, position));
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
            <div className={`${styles.scaleBar} ${styles.nounScale}`}>
              {clickedWordScore?.type === "noun" && (
                <div
                  className={`${styles.scoreMarker} ${styles.nounMarker}`}
                  style={{
                    left: `${getMarkerPosition(
                      clickedWordScore.score,
                      maxScores.noun
                    )}%`,
                  }}
                />
              )}
            </div>
          </div>
          <div className={styles.tickMarks}>
            {nounTicks.map((tick, index) => (
              <span key={`noun-${index}`} className={styles.tick}>
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
            <div className={`${styles.scaleBar} ${styles.verbScale}`}>
              {clickedWordScore?.type === "verb" && (
                <div
                  className={`${styles.scoreMarker} ${styles.verbMarker}`}
                  style={{
                    left: `${getMarkerPosition(
                      clickedWordScore.score,
                      maxScores.verb
                    )}%`,
                  }}
                />
              )}
            </div>
          </div>
          <div className={styles.tickMarks}>
            {verbTicks.map((tick, index) => (
              <span key={`verb-${index}`} className={styles.tick}>
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
