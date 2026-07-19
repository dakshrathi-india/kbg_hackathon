import {
  Atom,
  Github,
  Heart,
  Users,
} from "lucide-react";

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-brand">
          <Atom size={27} />

          <div>
            <strong>DELTA001</strong>
            <span>ADMET Intelligence Platform</span>
          </div>
        </div>

        <div className="creator-information">
          <Users size={18} />

          <div>
            <span>Created by</span>

            <strong>
              Daksh Rathi &amp; Priyanshu Barnwal
            </strong>
          </div>
        </div>

        <a
          href="https://github.com/dakshrathi-india/kbg_hackathon"
          target="_blank"
          rel="noreferrer"
        >
          <Github size={18} />
          View source code
        </a>
      </div>

      <div className="hackathon-note">
        Built with <Heart size={15} /> for the Computational
        Biology Hackathon.
      </div>

      <div className="disclaimer">
        This platform is a computational research prototype
        for preliminary screening. Predictions are not a
        replacement for laboratory, clinical or regulatory
        evaluation.
      </div>
    </footer>
  );
}

export default Footer;