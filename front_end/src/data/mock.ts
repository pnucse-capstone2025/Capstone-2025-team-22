export interface AnalyzedText {
  input: string;
  saveAt: string;
  noun: string[];
  verb: string[];
  adverb: string[];
  key_word: string[];
}

export interface WordCloudDatum {
  text: string;
  value: number;
}

export const mockData: AnalyzedText = {
  saveAt: "2025-08-05",
  input:
    "인터넷주소관리기관인 한국인터넷진흥원은 인터넷주소의 할당·등록 및 관련된 업무를 수행하는 정보통신망 이용촉진 및 정보보호 등에 관한 법률 제52조에 따른 한국인터넷진흥원(이하 ‘인터넷진흥원’이라 한다)과 인터넷주소운영사무소로부터 인터넷주소 관리업무를 위탁받은 법인 및 단체를 말한다.",
  noun: [
    "인터넷",
    "주소",
    "관리",
    "기관",
    "한국",
    "진흥원",
    "할당",
    "등록",
    "업무",
    "수행",
    "정보",
    "통신",
    "망",
    "이용",
    "촉진",
    "보호",
    "법률",
    "제52조",
    "운영",
    "사무소",
    "법인",
    "단체",
  ],
  verb: ["수행", "말한다", "받"],
  adverb: [],
  key_word: ["인터넷주소"],
};

export const mockDataNoKeywords: AnalyzedText = {
  saveAt: "2025-08-04",
  input: "안녕하세요. 오늘 날씨가 좋네요.",
  noun: [],
  verb: [],
  adverb: [],
  key_word: [],
};

// ✅ react-d3-cloud용 변환 유틸
const randomValue = () => Math.floor(Math.random() * 30) + 10;

export const nounWords: WordCloudDatum[] = mockData.noun.map((word) => ({
  text: word,
  value: randomValue(),
}));

export const verbWords: WordCloudDatum[] = mockData.verb.map((word) => ({
  text: word,
  value: randomValue(),
}));

export const adverbWords: WordCloudDatum[] = mockData.adverb.map((word) => ({
  text: word,
  value: randomValue(),
}));
