import { useState } from "react";
import {
  Atom,
  Github,
  Menu,
  X,
} from "lucide-react";

function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  const closeMenu = () => {
    setMenuOpen(false);
  };

  return (
    <header className="navbar">
      <div className="navbar-container">
        <a
          href="#home"
          className="brand"
          onClick={closeMenu}
        >
          <div className="brand-icon">
            <Atom size={25} />
          </div>

          <div>
            <span className="brand-name">DELTA001</span>
            <span className="brand-subtitle">
              ADMET Intelligence
            </span>
          </div>
        </a>

        <button
          type="button"
          className="mobile-menu-button"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle navigation menu"
        >
          {menuOpen ? <X /> : <Menu />}
        </button>

        <nav
          className={`nav-links ${
            menuOpen ? "nav-links-open" : ""
          }`}
        >
          <a href="#predict" onClick={closeMenu}>
            Predict
          </a>

          <a href="#performance" onClick={closeMenu}>
            Models
          </a>

          <a href="#methodology" onClick={closeMenu}>
            Methodology
          </a>

          <a
            href="https://github.com/dakshrathi-india/kbg_hackathon"
            target="_blank"
            rel="noreferrer"
            className="github-link"
            onClick={closeMenu}
          >
            <Github size={18} />
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}

export default Navbar;