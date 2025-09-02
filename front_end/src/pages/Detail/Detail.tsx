import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import styles from "./Detail.module.scss";
import WordCloudPage from "@/components/WordcloudSection/WordcloudSection";
import FloatingSearch from "@/components/FloatingSearch/FloatingSearch";
import Drawer from "@/components/Drawer/Drawer";
import { mockData, mockDataNoKeywords } from "@/data/mock";
import TextHighlighter from "@/pages/Detail/components/TextHighligher/TextHighlighter";

export const DetailPage = () => {
  const { id } = useParams<{ id: string }>();
  const currentData = id === "2" ? mockDataNoKeywords : mockData;
  const keywords = currentData.key_word;
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedKeyword, setSelectedKeyword] = useState<string | null>(
    keywords.length > 0 ? keywords[0] : null
  );
  const navigate = useNavigate();

  const toggleDrawer = () => setDrawerOpen((prev) => !prev);

  const handleSearch = (query: string) => {
    console.log("검색:", query);
    navigate("/detail/1");
  };

  return (
    <div className={styles.detailPage}>
      <button
        className={`${styles.hamburgerBtn} ${drawerOpen ? styles.open : ""}`}
        onClick={toggleDrawer}
      >
        ☰
      </button>

      <Drawer isOpen={drawerOpen} onToggle={toggleDrawer} />

      <div className={styles.detailWrapper}>
        <h3 className={styles.contentTitle}>키워드 분석 결과</h3>
        <div className={styles.keywordBox}>
          {keywords.length > 0 ? (
            <div className={styles.keywordButtons}>
              {keywords.map((kw: string, i: number) => (
                <button
                  key={i}
                  className={`${styles.keywordButton} ${
                    selectedKeyword === kw ? styles.selected : ""
                  }`}
                  onClick={() =>
                    setSelectedKeyword(selectedKeyword === kw ? null : kw)
                  }
                >
                  {kw}
                </button>
              ))}
            </div>
          ) : (
            <span className={styles.highlight}>추출된 키워드가 없습니다</span>
          )}
        </div>
        <div className={styles.textContent}>
          <TextHighlighter
            text={currentData.input}
            hoveredKeyword={selectedKeyword}
          />
        </div>
        <div className={styles.textContent}></div>
        <h3 className={styles.contentTitle}>품사별 분석 결과</h3>
        <div className={styles.detailContainer}>
          <WordCloudPage
            nounWords={currentData.noun}
            verbWords={currentData.verb}
            adverbWords={currentData.adverb}
          />
        </div>
      </div>
      <FloatingSearch onSearch={handleSearch} />
    </div>
  );
};
