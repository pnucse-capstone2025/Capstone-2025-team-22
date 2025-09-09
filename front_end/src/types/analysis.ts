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

export interface ClickedWordScore {
  score: number;
  type: "noun" | "verb";
}
