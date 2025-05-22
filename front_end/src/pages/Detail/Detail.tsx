import styles from './Detail.module.scss'
import { useParams } from 'react-router-dom'

export const DetailPage = () => {
   const { id } = useParams<{ id: string }>()

  return (
    <div className={styles.detailWrapper}>
      <div className = {styles.detailContainer}>
        <h1>검색 기록 상세 페이지</h1>
        <p>선택한 검색 기록 ID: {id}</p>
      </div>
    </div>
  )
}