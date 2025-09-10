// 문자열을 해시값으로 변환하는 함수
const hashString = (str: string): number => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // 32비트 정수로 변환
  }
  return Math.abs(hash);
};

// 단어를 기반으로 고정된 value 생성 (10-40 범위)
export const getWordValue = (word: string): number => {
  const hash = hashString(word);
  return (hash % 31) + 10; // 10~40 범위
};

// 단어를 기반으로 고정된 색상 인덱스 생성
export const getWordColorIndex = (word: string, colorsLength: number): number => {
  const hash = hashString(word);
  return hash % colorsLength;
};
