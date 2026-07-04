import { NavLink, Outlet } from "react-router-dom";

export default function Layout() {
  return (
    <div className="app">
      <header className="header">
        <div className="brand-row">
          <div className="brand">
            <span className="brand-mark">LTX-WS</span>
            <span className="brand-sub">Videofentanyl</span>
          </div>
          <nav className="main-nav" aria-label="Main">
            <NavLink to="/" end className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
              Generate
            </NavLink>
            <NavLink to="/train" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
              Train LoRA
            </NavLink>
          </nav>
        </div>
      </header>
      <Outlet />
    </div>
  );
}
