import { memo, useMemo } from "react";
import WordCloud from "react-d3-cloud";
import { mockData } from "@/data/mock";
import styles from "@/pages/Detail/sections/WordcloudSection/WordcloudSection.module.scss";

interface WordCloudDatum {
  text: string;
  value: number;
  color: string;
}

interface WordCloudSectionProps {
  words: string[];
  title: string;
}

const colors = ["#3EC1D3", "#23C8EF", "#FF165D", "#F472B6"];

const WordCloudSection = ({ words, title }: WordCloudSectionProps) => {
  if (!words || words.length < 1) {
    const partOfSpeech = title.includes("명사")
      ? "명사"
      : title.includes("동사")
      ? "동사"
      : "형용사";
    return (
      <div className={styles.wordCloudSection}>
        <h3 className={styles.title}>{title}</h3>
        <p className={styles.noresultmsg}>
          텍스트 분석 결과에 {partOfSpeech}가 없습니다.
        </p>
      </div>
    );
  }

  const formattedWords: WordCloudDatum[] = useMemo(() => {
    return words.map((word) => ({
      text: word,
      value: Math.floor(Math.random() * 30) + 10,
      color: colors[Math.floor(Math.random() * colors.length)],
    }));
  }, [words]);

  return (
    <div className={styles.wordCloudSection}>
      <h3 className={styles.title}>{title}</h3>
      <div className={styles.cloudContainer}>
        <WordCloud
          data={formattedWords}
          fontSize={(w: WordCloudDatum) => Math.log2(w.value) * 16}
          font="Pretendard"
          fill={(w: WordCloudDatum) => w.color}
          rotate={() => 0}
          padding={2}
          width={800}
          height={400}
        />
      </div>
    </div>
  );
};

interface WordCloudPageProps {
  nounWords?: string[];
  verbWords?: string[];
  adverbWords?: string[];
}

const WordCloudPage = ({
  nounWords = mockData.noun,
  verbWords = mockData.verb,
  adverbWords = mockData.adverb,
}: WordCloudPageProps) => {
  return (
    <div className={styles.wrapper}>
      <div className={styles.leftSection}>
        <WordCloudSection words={nounWords} title="명사 워드클라우드" />
      </div>
      <div className={styles.rightSection}>
        <WordCloudSection words={verbWords} title="동사 워드클라우드" />
        <WordCloudSection words={adverbWords} title="형용사 워드클라우드" />
      </div>
    </div>
  );
};

export default memo(WordCloudPage);
