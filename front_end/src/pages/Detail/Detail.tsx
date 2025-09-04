import { useEffect, useState } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import styles from "./Detail.module.scss";
import WordCloudPage from "@/components/WordcloudSection/WordcloudSection";
import FloatingSearch from "@/components/FloatingSearch/FloatingSearch";
import Drawer from "@/components/Drawer/Drawer";
//import { mockData, mockDataNoKeywords } from "@/data/mock";
import TextHighlighter from "@/pages/Detail/components/TextHighligher/TextHighlighter";

interface AnalysisResult {
  text: string;
  nouns: string[];
  verbs: string[];
  adjectives: string[];
  keywords: string[];
  noun_count: number;
  verb_count: number;
  adjective_count: number;
}

interface RecentResult {
  id: number;
  text: string;
  created_at: string;
  nouns: string[];
  verbs: string[];
  adjectives: string[];
  keywords: string[];
}

export const DetailPage = () => {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();

  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [recentResults, setRecentResults] = useState<RecentResult[]>([]);

  //const currentData = id === "2" ? mockDataNoKeywords : mockData;
  //const keywords = currentData.key_word;
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedKeyword, setSelectedKeyword] = useState<string | null>(null);
  const navigate = useNavigate();

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

    const fetchAnalysisResultById = async (analysisId: string) => {
      try {
        const response = await fetch(
          `http://localhost:8000/analysis_results/${analysisId}`
        );
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAnalysisData(data);
        if (data.keywords && data.keywords.length > 0) {
          setSelectedKeyword(data.keywords[0]);
        }
      } catch (error) {
        console.error(
          `ID ${analysisId}에 대한 분석 결과를 불러오는 중 오류 발생:`,
          error
        );
        setAnalysisData(null);
      }
    };

    fetchRecentResults();

    if (location.state && location.state.analysisResult) {
      setAnalysisData(location.state.analysisResult);
    } else if (id) {
      // URL에서 id를 가져와서 데이터를 fetch
      fetchAnalysisResultById(id);
    } else {
      console.error("분석 결과 데이터가 없습니다.");
    }
  }, [location.state, drawerOpen, id]);

  const keywords = analysisData?.keywords || [];

  const toggleDrawer = () => setDrawerOpen((prev) => !prev);

  const handleSearch = async (query: string) => {
    console.log("분석 텍스트:", query);
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

      if (!result.keywords || result.keywords.length === 0) {
        console.warn("키워드가 추출되지 않았습니다.");
        setAnalysisData(result);
        setSelectedKeyword(null);
        navigate(`/detail/${result.id}`, { state: { analysisResult: result } });
        return;
      }

      setAnalysisData(result);
      setSelectedKeyword(result.keywords[0]);
      navigate(`/detail/${result.id}`, { state: { analysisResult: result } });
    } catch (error) {
      console.error("분석 중 오류 발생:", error);
    }
  };

  return (
    <div className={styles.detailPage}>
      <button
        className={`${styles.hamburgerBtn} ${drawerOpen ? styles.open : ""}`}
        onClick={toggleDrawer}
      >
        ☰
      </button>

      <Drawer
        isOpen={drawerOpen}
        onToggle={toggleDrawer}
        recentResults={recentResults}
      />

      <div className={styles.detailWrapper}>
        <h3 className={styles.contentTitle}>키워드 분석 결과</h3>
        <p className={styles.contentDescription}>
          키워드를 클릭하시면 해당 키워드가 추출된 이유를 확인하실 수 있습니다.
        </p>
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
            text={analysisData?.text || ""}
            hoveredKeyword={selectedKeyword}
          />
        </div>
        <div className={styles.gradationScale}></div>
        <h3 className={styles.contentTitle}>품사별 분석 결과</h3>
        <p className={styles.contentDescription}>
          품사 분석 과정에는 MeCab 라이브러리를 사용하였습니다.
        </p>
        <div className={styles.detailContainer}>
          <WordCloudPage
            nounWords={analysisData?.nouns || []}
            verbWords={analysisData?.verbs || []}
            adverbWords={analysisData?.adjectives || []}
          />
        </div>
      </div>
      <FloatingSearch onSearch={handleSearch} />
    </div>
  );
};
