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
    "인터넷주소관리기관은 「정보통신망 이용촉진 및 정보보호 등에 관한 법률」 제52조를 근거로 하여 설립된 기관으로, 우리나라 인터넷주소 체계의 안정적 운영과 효율적 관리를 위해 중요한 역할을 맡고 있다. 대표적으로 한국인터넷진흥원이 이에 해당하며, 인터넷주소의 등록, 할당, 유지 관리와 같은 실무를 직접 수행한다. 또한 인터넷주소운영사무소로부터 업무를 위탁받아 운영하는 법인이나 단체도 인터넷주소관리기관의 범주에 포함된다. 이 기관은 단순히 주소를 관리하는 차원을 넘어, 인터넷 이용자들의 권익 보호와 인터넷 환경의 투명성 확보에도 기여한다. 주소 분배 과정에서 공정성과 효율성을 확보하고, 국제 인터넷주소 관리 기구와의 협력에도 참여하여 글로벌 표준과 정책 변화에 적극 대응한다. 이를 통해 국내 인터넷 생태계가 국제적 흐름에 부합하도록 조정하는 역할을 한다. 또한 인터넷주소관리기관은 사이버 보안 측면에서도 매우 중요한 기능을 수행한다. 안정적인 주소 체계는 안전한 인터넷 환경의 기초가 되며, 이를 통해 기업과 개인이 안심하고 인터넷을 활용할 수 있는 기반을 마련한다. 아울러 정책 개발, 제도 개선, 기술 지원 등 다양한 활동을 통해 국가 차원의 정보통신 인프라 발전을 이끌고 있다. 결국 인터넷주소관리기관은 단순한 행정적 주체가 아니라, 인터넷 생태계의 지속 가능한 성장과 국민의 안전한 디지털 생활을 보장하는 핵심 기관으로서, 국가 정보통신 발전 전략의 중추적 역할을 수행한다고 할 수 있다.",
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
  key_word: ["인터넷주소", "정보보호", "디지털"],
};

export const mockDataNoKeywords: AnalyzedText = {
  saveAt: "2025-08-04",
  input: "안녕하세요. 오늘 날씨가 좋네요.",
  noun: [],
  verb: [],
  adverb: [],
  key_word: [],
};

export const mockDataAnalysis = {
  keywordList: ["인터넷주소", "정보보호", "디지털"],
  keywordAnalysis: [
    {
      keyword: "인터넷주소",
      contributions: [
        { start: 0, end: 10, score: 10 },
        { start: 66, end: 71, score: 9 },
        { start: 134, end: 139, score: 8 },
        { start: 263, end: 268, score: 7 },
        { start: 367, end: 372, score: 6 },
        { start: 420, end: 425, score: 2 },
        { start: 500, end: 505, score: 4 },
        { start: 600, end: 605, score: 3 },
        { start: 700, end: 705, score: 8 },
      ],
    },
    {
      keyword: "정보보호",
      contributions: [
        { start: 25, end: 29, score: 3 },
        { start: 150, end: 154, score: 9 },
        { start: 250, end: 254, score: 8 },
        { start: 350, end: 354, score: 7 },
        { start: 450, end: 460, score: 2 },
        { start: 550, end: 554, score: 5 },
        { start: 650, end: 654, score: 4 },
      ],
    },
    {
      keyword: "디지털",
      contributions: [
        { start: 816, end: 819, score: 10 },
        { start: 100, end: 103, score: 9 },
        { start: 200, end: 203, score: 8 },
        { start: 300, end: 303, score: 7 },
        { start: 400, end: 403, score: 6 },
        { start: 500, end: 503, score: 5 },
      ],
    },
  ],
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
