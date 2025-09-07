import { useState, useEffect } from "react";
import { useNavigate, Outlet } from "react-router-dom";
import SearchBar from "@/components/SearchBar/SearchBar";
import Drawer from "@/components/Drawer/Drawer";
import styles from "./MainLayout.module.scss";

interface RecentResult {
  id: number;
  text: string;
  created_at: string;
  nouns: string[];
  verbs: string[];
  adjectives: string[];
  keywords: string[];
}

export function MainLayout() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [recentResults, setRecentResults] = useState<RecentResult[]>([]);
  const navigate = useNavigate();

  const toggleDrawer = () => setDrawerOpen((prev) => !prev);

  useEffect(() => {
    const fetchRecentResults = async () => {
      try {
        const response = await fetch("http://localhost:8000/recent_results");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setRecentResults(data.results || []);
      } catch (error) {
        console.error("최근 검색 기록을 불러오는 중 오류 발생:", error);
      }
    };

    fetchRecentResults();
  }, [drawerOpen]);

  const handleSearch = async (query: string) => {
    console.log("분석:", query);
    try {
      const response = await fetch("http://localhost:8000/extract_keywords", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          text: query,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log("분석 결과:", result);

      setRecentResults((prevResults) =>
        [
          {
            id: result.id,
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
      <Outlet />
    </div>
  );
}
