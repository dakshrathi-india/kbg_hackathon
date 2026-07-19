import {
  Atom,
  BrainCircuit,
  CheckCircle2,
  Fingerprint,
  FlaskConical,
  GitMerge,
  ScanSearch,
} from "lucide-react";

const STEPS = [
  {
    number: "01",
    icon: FlaskConical,
    title: "SMILES Input",
    text: "The user enters the molecular structure as a SMILES string.",
  },
  {
    number: "02",
    icon: ScanSearch,
    title: "Molecule Validation",
    text: "RDKit validates the molecule and converts it into canonical SMILES.",
  },
  {
    number: "03",
    icon: Fingerprint,
    title: "Feature Generation",
    text: "Morgan fingerprints and physicochemical molecular descriptors are generated.",
  },
  {
    number: "04",
    icon: BrainCircuit,
    title: "Seed-Averaged Models",
    text: "Extra Trees, XGBoost, CatBoost and LightGBM produce predictions using multiple random seeds.",
  },
  {
    number: "05",
    icon: GitMerge,
    title: "Weighted Ensemble",
    text: "The seed-averaged predictions are combined using endpoint-specific ensemble weights.",
  },
  {
    number: "06",
    icon: Atom,
    title: "Molecular Profile",
    text: "The system returns five ADMET predictions along with aqueous solubility.",
  },
];

function MethodologySection() {
  return (
    <section
      id="methodology"
      className="methodology-section page-section"
    >
      <div className="section-heading">
        <div className="section-label">
          <BrainCircuit size={18} />
          End-to-End Methodology
        </div>

        <h2>From molecular structure to predictions</h2>

        <p>
          A transparent machine-learning pipeline designed
          for rapid early-stage molecular screening.
        </p>
      </div>

      <div className="methodology-grid">
        {STEPS.map(
          ({ number, icon: Icon, title, text }) => (
            <article
              className="methodology-card glass-card"
              key={number}
            >
              <div className="method-number">
                {number}
              </div>

              <div className="method-icon">
                <Icon size={25} />
              </div>

              <h3>{title}</h3>

              <p>{text}</p>
            </article>
          ),
        )}
      </div>

      <div className="technical-summary glass-card">
        <div>
          <CheckCircle2 size={23} />

          <div>
            <strong>Scientifically grounded</strong>

            <p>
              Invalid molecules are rejected, preprocessing
              is reproduced exactly and every endpoint uses
              molecular scaffold-aware data splitting.
            </p>
          </div>
        </div>

        <div>
          <CheckCircle2 size={23} />

          <div>
            <strong>Inference only</strong>

            <p>
              The deployed application loads already-trained
              models. User requests never retrain or modify
              the machine-learning models.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default MethodologySection;