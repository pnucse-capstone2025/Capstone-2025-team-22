export interface WordCloudItem {
  text: string;
  value: number;
  color: string;
}

export interface WordCloudSectionProps {
  words: string[];
  title: string;
}

export interface WordCloudPageProps {
  nounWords?: string[];
  verbWords?: string[];
  adverbWords?: string[];
} 