import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import styles from "./Detail.module.scss";
import WordCloudPage from "@/components/WordcloudSection/WordcloudSection";
import FloatingSearch from "@/components/FloatingSearch/FloatingSearch";
import Drawer from "@/components/Drawer/Drawer";
import { mockData, mockDataNoKeywords } from "@/data/mock";

export const DetailPage = () => {
  const { id } = useParams<{ id: string }>();
  const currentData = id === "2" ? mockDataNoKeywords : mockData;
  const keywords = currentData.key_word;
  const [drawerOpen, setDrawerOpen] = useState(false);
  const navigate = useNavigate();

  const toggleDrawer = () => setDrawerOpen((prev) => !prev);

  const handleSearch = (query: string) => {
    // 검색 로직 구현
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
        <div className={styles.keywordBox}>
          <p className={styles.keywordContent}>
            {keywords.length > 0 ? (
              <>
                키워드 :{" "}
                {keywords.map((kw: string, i: number) => (
                  <span key={i} className={styles.highlight}>
                    {kw}
                    {i < keywords.length - 1 && ",\u00A0"}
                  </span>
                ))}{" "}
              </>
            ) : (
              <span className={styles.highlight}>추출된 키워드가 없습니다</span>
            )}
          </p>
        </div>
        <div className={styles.textContent}>
          <p>{currentData.input}</p>
        </div>
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
