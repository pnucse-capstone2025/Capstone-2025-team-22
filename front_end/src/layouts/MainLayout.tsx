import { useState } from 'react'
import { useNavigate, Outlet } from 'react-router-dom'
import SearchBar from '../components/SearchBar/SearchBar'
import styles from './MainLayout.module.scss'

export function MainLayout() {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const navigate = useNavigate()

  const toggleDrawer = () => setDrawerOpen(prev => !prev)
  const handleCardClick = (id: number) => {
    navigate(`/detail/${id}`)
  }

  return (
    <div className={styles.layoutWrapper}>
      <button
        className={`${styles.hamburgerBtn} ${drawerOpen ? styles.open : ''}`}
        onClick={toggleDrawer}
      >
        ☰
      </button>

      <SearchBar className={styles.searchBar} onSearch={() => {}} />

      <aside className={`${styles.drawer} ${drawerOpen ? styles.open : ''}`}>
        {[...Array(15)].map((_, i) => (
          <div
            key={i}
            className={styles.drawerBlock}
            onClick={() => handleCardClick(i + 1)}
            role="button"
            tabIndex={0}
          >
            과거 검색 기록 {i + 1}
          </div>
        ))}
      </aside>
      {drawerOpen && <div className={styles.backdrop} onClick={toggleDrawer} />}
      <Outlet />
    </div>
  )
}
