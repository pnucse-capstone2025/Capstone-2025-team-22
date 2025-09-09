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

  // 실제 하이라이트 색상 계산 (TextHighlighter와 동일한 로직)
  const getActualHighlightColor = (score: number, type: 'noun' | 'verb') => {
    const opacity = Math.min(score, 1) * 0.8;
    return type === 'noun'
      ? `rgba(220, 53, 69, ${opacity})`
      : `rgba(30, 144, 255, ${opacity})`;
  };

  // 동적 그라디언트 생성
  const generateScaleGradient = (maxScore: number, type: 'noun' | 'verb') => {
    const baseColor = type === 'noun' ? '220, 53, 69' : '30, 144, 255';
    const steps = [];
    
    for (let i = 0; i <= 4; i++) {
      const ratio = i / 4; // 0, 0.25, 0.5, 0.75, 1
      const score = maxScore * ratio;
      const opacity = Math.min(score, 1) * 0.8;
      steps.push(`rgba(${baseColor}, ${opacity}) ${ratio * 100}%`);
    }
    
    return `linear-gradient(to right, ${steps.join(', ')})`;
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
                background: generateScaleGradient(maxScores.noun, 'noun')
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
                    backgroundColor: getActualHighlightColor(clickedWordScore.score, 'noun'),
                    border: '2px solid #ffffff',
                    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
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
            <div 
              className={styles.scaleBar}
              style={{ 
                background: generateScaleGradient(maxScores.verb, 'verb')
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
                    backgroundColor: getActualHighlightColor(clickedWordScore.score, 'verb'),
                    border: '2px solid #ffffff',
                    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
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
