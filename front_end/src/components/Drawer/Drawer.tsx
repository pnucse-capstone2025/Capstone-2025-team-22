import { useNavigate } from "react-router-dom";
import type { DrawerProps } from "@/types";
import styles from "./Drawer.module.scss";

export default function Drawer({
  isOpen,
  onToggle,
  recentResults,
}: DrawerProps) {
  const navigate = useNavigate();

  const handleCardClick = (id: number) => {
    navigate(`/detail/${id}`);
  };

  const displayResults = (recentResults || []).slice(0, 15);

  const paddedResults = [
    ...displayResults,
    ...Array(15 - displayResults.length).fill(null),
  ];

  return (
    <>
      <aside className={`${styles.drawer} ${isOpen ? styles.open : ""}`}>
        {paddedResults.map((result, i) => (
          <div
            key={result ? result.id : `empty-${i}`}
            className={styles.drawerBlock}
            onClick={() => result && handleCardClick(result.id)}
            role="button"
            tabIndex={result ? 0 : -1}
          >
            <div className={styles.drawerText}>
              {result
                ? result.keywords.length > 0
                  ? result.keywords.join(", ")
                  : "키워드 없음"
                : ""}
            </div>
            <div className={styles.drawerDate}>
              {result ? result.created_at : ""}
            </div>
          </div>
        ))}
      </aside>
      {isOpen && <div className={styles.backdrop} onClick={onToggle} />}
    </>
  );
}
