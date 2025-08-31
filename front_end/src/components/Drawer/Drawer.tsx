import { useNavigate } from "react-router-dom";
import styles from "./Drawer.module.scss";

interface DrawerProps {
  isOpen: boolean;
  onToggle: () => void;
}

export default function Drawer({ isOpen, onToggle }: DrawerProps) {
  const navigate = useNavigate();

  const handleCardClick = (id: number) => {
    navigate(`/detail/${id}`);
  };

  return (
    <>
      <aside className={`${styles.drawer} ${isOpen ? styles.open : ""}`}>
        {[...Array(15)].map((_, i) => (
          <div
            key={i}
            className={styles.drawerBlock}
            onClick={() => handleCardClick(i + 1)}
            role="button"
            tabIndex={0}
          >
            <div className={styles.drawerText}>과거 검색 기록 {i + 1}</div>
            <div className={styles.drawerDate}>2025-08-05</div>
          </div>
        ))}
      </aside>
      {isOpen && <div className={styles.backdrop} onClick={onToggle} />}
    </>
  );
}
