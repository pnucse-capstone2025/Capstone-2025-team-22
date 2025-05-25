import { FormEvent, useRef, useState, useEffect } from 'react'
import styles from './SearchBar.module.scss'

interface SearchBarProps {
  onSearch: (query: string) => void
  className?: string 
}

export default function SearchBar({ onSearch }: SearchBarProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [minHeight, setMinHeight] = useState<number>(0)
  const [showSubButtons, setShowSubButtons] = useState(false)

  useEffect(() => {
    const ta = textareaRef.current
    if (ta) {
      setMinHeight(ta.clientHeight)
    }
  }, [])

  const handleResizeHeight = () => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    const newHeight = Math.max(ta.scrollHeight, minHeight)
    ta.style.height = `${newHeight}px`
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value)
    handleResizeHeight()
  }

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    onSearch(value.trim())
    setShowSubButtons(prev => !prev)
  }

  return (
    <form className={styles.searchBar} onSubmit={handleSubmit}>
      <textarea
        ref={textareaRef}
        className={styles.searchInput}
        placeholder="분석을 원하시는 텍스트를 넣어주세요"
        value={value}
        onChange={handleChange}
        rows={1}
      />
       <button type="submit" className={styles.mainButton}>분석</button>

      {showSubButtons && (
        <div className={styles.subButtonGroup}>
          <button type="button" className={styles.subButton}>명사</button>
          <button type="button" className={styles.subButton}>형용사</button>
          <button type="button" className={styles.subButton}>동사</button>
        </div>
      )}
    </form>
  )
}
