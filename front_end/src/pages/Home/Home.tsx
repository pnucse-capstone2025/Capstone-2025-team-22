import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import SearchBar from "@/components/SearchBar/SearchBar";
import Drawer from "@/components/Drawer/Drawer";
import styles from "./Home.module.scss";
import type { RecentResult } from "@/types";
import { getRecentResults } from "@/api/services/results";
import { extractKeywords } from "@/api/services/analysis";

export const HomePage = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [recentResults, setRecentResults] = useState<RecentResult[]>([]);
  const navigate = useNavigate();

  const toggleDrawer = () => setDrawerOpen((prev) => !prev);

  useEffect(() => {
    const fetchRecentResults = async () => {
      try {
        const results = await getRecentResults();
        setRecentResults(results);
      } catch (error) {
        console.error("최근 검색 기록을 불러오는 중 오류 발생:", error);
      }
    };

    fetchRecentResults();
  }, [drawerOpen]);

  const handleSearch = async (query: string) => {
    console.log("분석:", query);
    try {
      const result = await extractKeywords(query);
      console.log("분석 결과:", result);

      setRecentResults((prevResults) =>
        [
          {
            id: Date.now(), // 임시 ID 생성
            text: result.text,
            created_at: new Date().toISOString().slice(0, 10),
            nouns: result.nouns,
            verbs: result.verbs,
            adjectives: result.adjectives,
            keywords: result.keywords,
          },
          ...prevResults,
        ].slice(0, 15)
      );

      navigate("/detail/1", { state: { analysisResult: result } });
    } catch (error) {
      console.error("분석 중 오류 발생:", error);
    }
  };

  return (
    <div className={styles.layoutWrapper}>
      <button
        className={`${styles.hamburgerBtn} ${drawerOpen ? styles.open : ""}`}
        onClick={toggleDrawer}
      >
        ☰
      </button>
      <div className={styles.wrapper}>
        <SearchBar className={styles.searchBar} onSearch={handleSearch} />
      </div>
      <Drawer
        isOpen={drawerOpen}
        onToggle={toggleDrawer}
        recentResults={recentResults}
      />
    </div>
  );
};
