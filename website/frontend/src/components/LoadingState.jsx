import {
  Atom,
  BrainCircuit,
  Fingerprint,
} from "lucide-react";

function LoadingState() {
  return (
    <section className="loading-section">
      <div className="loading-card glass-card">
        <div className="loading-atom">
          <Atom size={42} />
        </div>

        <h3>Analyzing molecular structure</h3>

        <p>
          Generating molecular fingerprints and running the
          ensemble models.
        </p>

        <div className="loading-steps">
          <div>
            <Atom size={19} />
            Validating SMILES
          </div>

          <div>
            <Fingerprint size={19} />
            Building features
          </div>

          <div>
            <BrainCircuit size={19} />
            Running models
          </div>
        </div>

        <div className="loading-progress">
          <span />
        </div>
      </div>
    </section>
  );
}

export default LoadingState;