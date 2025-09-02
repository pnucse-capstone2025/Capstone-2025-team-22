import { useRef, useEffect, useState } from "react";
import type { FormEvent, ChangeEvent } from "react";
import styles from "@/components/SearchBar/SearchBar.module.scss";

interface SearchBarProps {
  onSearch: (query: string) => void;
  className?: string;
}

export default function SearchBar({ onSearch, className }: SearchBarProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [minHeight, setMinHeight] = useState<number>(0);

  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      setMinHeight(ta.clientHeight);
    }
  }, []);

  const handleResizeHeight = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    const newHeight = Math.max(ta.scrollHeight, minHeight);
    ta.style.height = `${newHeight}px`;
  };

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    handleResizeHeight();
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onSearch(value.trim());
  };

  return (
    <div className={styles.searchBarContainer}>
      <form className={`${styles.searchBar} ${className || ''}`} onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className={styles.searchInput}
          placeholder="분석을 원하시는 텍스트를 넣어주세요"
          value={value}
          onChange={handleChange}
          rows={1}
        />
      </form>
      <button 
        type="button" 
        className={styles.analyzeButton}
        onClick={() => onSearch(value.trim())}
      >
        분석하기
      </button>
    </div>
  );
}
