import styles from "./Detail.module.scss";
import { useParams } from "react-router-dom";
import WordCloudPage from "@/components/WordcloudSection/WordcloudSection";
import { mockData } from "@/data/mock";

export const DetailPage = () => {
  const { id } = useParams<{ id: string }>();
  const keywords = mockData.key_word;

  return (
    <div className={styles.detailWrapper}>
      <p>선택한 검색 기록 ID: {id}</p>
      <p>
        텍스트의 키워드는{" "}
        {keywords.length > 0 ? (
          keywords.map((kw: string, i: number) => (
            <span key={i} className={styles.highlight}>
              {kw}
              {i < keywords.length - 1 && ",\u00A0"}
            </span>
          ))
        ) : (
          <span className={styles.highlight}>키워드가 없습니다</span>
        )}{" "}
        입니다
      </p>
      <div className={styles.detailContainer}>
        <WordCloudPage />
      </div>
    </div>
  );
};
