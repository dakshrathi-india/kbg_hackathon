import {
  ArrowRight,
  Atom,
  BrainCircuit,
  Database,
  Fingerprint,
  GitMerge,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";

const FEATURES = [
  {
    icon: Fingerprint,
    text: "Morgan Fingerprints",
  },
  {
    icon: Database,
    text: "RDKit Descriptors",
  },
  {
    icon: BrainCircuit,
    text: "4-Model Ensemble",
  },
  {
    icon: ShieldCheck,
    text: "Scaffold Split",
  },
];

const PROPERTY_NODES = [
  {
    code: "A",
    title: "Absorption",
  },
  {
    code: "D",
    title: "Distribution",
  },
  {
    code: "M",
    title: "Metabolism",
  },
  {
    code: "E",
    title: "Excretion",
  },
  {
    code: "T",
    title: "Toxicity",
  },
  {
    code: "S",
    title: "Solubility",
  },
];

function Hero() {
  const moveToPredictor = () => {
    document
      .getElementById("predict")
      ?.scrollIntoView({
        behavior: "smooth",
      });
  };

  const updateMouseGlow = (event) => {
    const bounds =
      event.currentTarget.getBoundingClientRect();

    const mouseX =
      event.clientX - bounds.left;

    const mouseY =
      event.clientY - bounds.top;

    event.currentTarget.style.setProperty(
      "--mouse-x",
      `${mouseX}px`,
    );

    event.currentTarget.style.setProperty(
      "--mouse-y",
      `${mouseY}px`,
    );
  };

  return (
    <section
      id="home"
      className="hero hero-v2"
      onMouseMove={updateMouseGlow}
    >
      <div className="hero-mouse-glow" />
      <div className="hero-grid-overlay" />

      <div className="hero-grid-v2">
        <div className="hero-copy-v2">
          <div className="hero-label">
            <Sparkles size={17} />
            AI-Powered Molecular Intelligence
          </div>

          <h1 className="hero-title-v2">
            Decode a molecule.
            <span>Predict its ADMET profile.</span>
          </h1>

          <p className="hero-description-v2">
            Enter a molecular SMILES string and instantly
            generate predictions for absorption,
            distribution, metabolism, excretion, toxicity
            and aqueous solubility.
          </p>

          <div className="hero-actions">
            <button
              type="button"
              className="primary-button hero-primary-button"
              onClick={moveToPredictor}
            >
              <Zap size={20} />
              Start Molecular Analysis
              <ArrowRight size={20} />
            </button>

            <a
              href="#methodology"
              className="secondary-button"
            >
              Explore the Pipeline
            </a>
          </div>

          <div className="feature-badges">
            {FEATURES.map(
              ({ icon: Icon, text }) => (
                <div
                  className="feature-badge"
                  key={text}
                >
                  <Icon size={17} />
                  <span>{text}</span>
                </div>
              ),
            )}
          </div>

          <div className="hero-statistics">
            <div>
              <strong>6</strong>
              <span>Predicted properties</span>
            </div>

            <div>
              <strong>2,060</strong>
              <span>Molecular features</span>
            </div>

            <div>
              <strong>4 × 3</strong>
              <span>Models and seeds</span>
            </div>
          </div>
        </div>

        <div className="hero-visual-v2">
          <div className="hero-visual-glow" />

          <div className="hero-console glass-card">
            <div className="hero-console-header">
              <div>
                <span className="console-eyebrow">
                  Live molecular intelligence
                </span>

                <h3>ADMET prediction engine</h3>
              </div>

              <div className="system-status">
                <span />
                System ready
              </div>
            </div>

            <div className="molecular-stage">
              <div className="molecular-ring ring-large" />
              <div className="molecular-ring ring-small" />

              <div className="central-molecule">
                <Atom size={56} />
                <span>SMILES</span>
              </div>

              {PROPERTY_NODES.map(
                ({ code, title }, index) => (
                  <div
                    key={code}
                    className={`property-node property-node-${
                      index + 1
                    }`}
                  >
                    <strong>{code}</strong>
                    <span>{title}</span>
                  </div>
                ),
              )}
            </div>

            <div className="hero-pipeline">
              <div>
                <Fingerprint size={18} />
                <span>Feature extraction</span>
              </div>

              <ArrowRight size={16} />

              <div>
                <BrainCircuit size={18} />
                <span>Seed averaging</span>
              </div>

              <ArrowRight size={16} />

              <div>
                <GitMerge size={18} />
                <span>Final ensemble</span>
              </div>
            </div>

            <div className="console-footer">
              <div>
                <span>Input</span>
                <strong>Molecular SMILES</strong>
              </div>

              <div>
                <span>Output</span>
                <strong>Complete screening profile</strong>
              </div>
            </div>
          </div>
        </div>
      </div>

      <button
        type="button"
        className="hero-scroll-indicator"
        onClick={moveToPredictor}
      >
        <span>Explore predictor</span>
        <div />
      </button>
    </section>
  );
}

export default Hero;