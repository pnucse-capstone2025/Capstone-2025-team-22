import { useState } from "react";
import { useNavigate, Outlet } from "react-router-dom";
import SearchBar from "@/components/SearchBar/SearchBar";
import Drawer from "@/components/Drawer/Drawer";
import styles from "./MainLayout.module.scss";

export function MainLayout() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const navigate = useNavigate();

  const toggleDrawer = () => setDrawerOpen((prev) => !prev);

  const handleSearch = (query: string) => {
    // 분석 로직 실행 후 최신 기록으로 이동
    console.log("분석:", query);
    navigate('/detail/1');
  };

  return (
    <div className={styles.layoutWrapper}>
      <button
        className={`${styles.hamburgerBtn} ${drawerOpen ? styles.open : ""}`}
        onClick={toggleDrawer}
      >
        ☰
      </button>
      <div className={styles.wrapper}>
        <SearchBar className={styles.searchBar} onSearch={handleSearch} />
      </div>
      <Drawer isOpen={drawerOpen} onToggle={toggleDrawer} />
      <Outlet />
    </div>
  );
}
