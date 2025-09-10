import { useEffect, useState } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import styles from "@/pages/Detail/Detail.module.scss";
import WordCloudPage from "@/pages/Detail/Sections/WordcloudSection/WordcloudSection";
import FloatingSearch from "@/components/FloatingSearch/FloatingSearch";
import Drawer from "@/components/Drawer/Drawer";
import KeywordAnalysisSection from "@/pages/Detail/Sections/KeywordAnalysisSection/KeywordAnalysisSection";
import type { AnalysisResult, RecentResult } from "@/types";
import { getRecentResults } from "@/api/services/results";
import { getAnalysisResult, extractKeywords } from "@/api/services/analysis";

const convertKeywordToString = (keyword: any): string => {
  if (typeof keyword === "string") return keyword;
  if (typeof keyword === "object" && keyword && "keyword" in keyword) {
    return keyword.keyword;
  }
  return String(keyword);
};

export const DetailPage = () => {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();

  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [recentResults, setRecentResults] = useState<RecentResult[]>([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedKeyword, setSelectedKeyword] = useState<string | null>(null);
  const navigate = useNavigate();

  const [clickedWordScore, setClickedWordScore] = useState<{
    score: number;
    type: "noun" | "verb";
  } | null>(null);

  useEffect(() => {
    const fetchRecentResults = async () => {
      try {
        const results = await getRecentResults();
        setRecentResults(results);
      } catch (error) {
        console.error("최근 검색 기록을 불러오는 중 오류 발생:", error);
      }
    };

    const fetchAnalysisResultById = async (analysisId: string) => {
      try {
        const data = await getAnalysisResult(analysisId);
        setAnalysisData(data);
        if (data.keywords && data.keywords.length > 0) {
          const firstKeyword = data.keywords[0];
          const keywordString = convertKeywordToString(firstKeyword);
          setSelectedKeyword(keywordString);

          setTimeout(() => {
            if (data.attention_result && data.attention_result[keywordString]) {
              const keywordData = data.attention_result[keywordString];
              const allWords: Array<{
                score: number;
                type: "noun" | "verb";
                start: number;
              }> = [];

              Object.values(keywordData.nouns || {}).forEach((item: any) => {
                allWords.push({
                  score: item.score,
                  type: "noun",
                  start: item.start,
                });
              });
              Object.values(keywordData.verbs || {}).forEach((item: any) => {
                allWords.push({
                  score: item.score,
                  type: "verb",
                  start: item.start,
                });
              });

              allWords.sort((a, b) => a.start - b.start);

              if (allWords.length > 0) {
                const firstWord = allWords[0];
                handleWordClick(firstWord.score, firstWord.type);
              }
            }
          }, 100);
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
      const data = location.state.analysisResult;
      if (data.keywords && data.keywords.length > 0) {
        const firstKeyword = data.keywords[0];
        const keywordString = convertKeywordToString(firstKeyword);
        setSelectedKeyword(keywordString);

        setTimeout(() => {
          if (data.attention_result && data.attention_result[keywordString]) {
            const keywordData = data.attention_result[keywordString];
            const allWords: Array<{
              score: number;
              type: "noun" | "verb";
              start: number;
            }> = [];

            Object.values(keywordData.nouns || {}).forEach((item: any) => {
              allWords.push({
                score: item.score,
                type: "noun",
                start: item.start,
              });
            });
            Object.values(keywordData.verbs || {}).forEach((item: any) => {
              allWords.push({
                score: item.score,
                type: "verb",
                start: item.start,
              });
            });

            allWords.sort((a, b) => a.start - b.start);

            if (allWords.length > 0) {
              const firstWord = allWords[0];
              handleWordClick(firstWord.score, firstWord.type);
            }
          }
        }, 100);
      }
    } else if (id) {
      fetchAnalysisResultById(id);
    } else {
      console.error("분석 결과 데이터가 없습니다.");
    }
  }, [location.state, drawerOpen, id]);

  const toggleDrawer = () => setDrawerOpen((prev) => !prev);

  const handleSearch = async (query: string) => {
    console.log("분석 텍스트:", query);
    try {
      const result = await extractKeywords(query);
      console.log("분석 결과:", result);

      if (!result.keywords || result.keywords.length === 0) {
        console.warn("키워드가 추출되지 않았습니다.");
        setAnalysisData(result);
        setSelectedKeyword(null);
        navigate(`/detail/${Date.now()}`, {
          state: { analysisResult: result },
        });
        return;
      }

      setAnalysisData(result);
      const firstKeyword = result.keywords[0];
      const keywordString = convertKeywordToString(firstKeyword);
      setSelectedKeyword(keywordString);
      navigate(`/detail/${Date.now()}`, { state: { analysisResult: result } });
    } catch (error) {
      console.error("분석 중 오류 발생:", error);
    }
  };

  const handleWordClick = (score: number, type: "noun" | "verb") => {
    console.log(`클릭된 단어 정보 - 타입: ${type}, 점수: ${score}`);
    setClickedWordScore({ score, type });
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
        <KeywordAnalysisSection
          analysisData={analysisData}
          selectedKeyword={selectedKeyword}
          clickedWordScore={clickedWordScore}
          onKeywordSelect={setSelectedKeyword}
          onWordClick={handleWordClick}
        />
        <h3 className={styles.contentTitle}>품사별 분석 결과</h3>
        <p className={styles.contentDescription}>
          품사 분석 과정에는 Mecab 라이브러리를 사용하였습니다.
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
