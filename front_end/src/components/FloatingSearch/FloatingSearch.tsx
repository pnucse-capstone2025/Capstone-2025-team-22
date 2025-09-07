import { useState, useRef, useEffect } from "react";
import { LuTextSearch } from "react-icons/lu";
import styles from "./FloatingSearch.module.scss";

interface FloatingSearchProps {
  onSearch: (query: string) => void;
}

export default function FloatingSearch({ onSearch }: FloatingSearchProps) {
  const [isOpen, setIsOpen] = useState(false);
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

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    handleResizeHeight();
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(value.trim());
    setValue("");
    setIsOpen(false);
  };

  const toggleSearch = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setTimeout(() => {
        textareaRef.current?.focus();
      }, 100);
    }
  };

  return (
    <div className={styles.floatingSearch}>
      {isOpen && (
        <div className={styles.searchContainer}>
          <form onSubmit={handleSubmit} className={styles.searchForm}>
            <textarea
              ref={textareaRef}
              className={styles.searchInput}
              placeholder="분석을 원하시는 텍스트를 넣어주세요"
              value={value}
              onChange={handleChange}
              rows={1}
            />
            <button type="submit" className={styles.searchButton}>
              텍스트 분석하기
            </button>
          </form>
        </div>
      )}
      <button
        className={`${styles.searchIcon} ${isOpen ? styles.active : ""}`}
        onClick={toggleSearch}
        type="button"
      >
        <LuTextSearch size={20} />
      </button>
    </div>
  );
} 