import {
  Binary,
  Eraser,
  FlaskConical,
  ScanLine,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

import { EXAMPLE_MOLECULES } from "../data/examples";

function SmilesInput({
  smiles,
  setSmiles,
  onAnalyze,
  loading,
  error,
}) {
  const handleSubmit = (event) => {
    event.preventDefault();
    onAnalyze();
  };

  return (
    <section id="predict" className="predict-section">
      <div className="section-heading">
        <div className="section-label">
          <FlaskConical size={18} />
          Molecular Analysis Workspace
        </div>

        <h2>Analyze your molecule</h2>

        <p>
          Paste a valid SMILES string to generate a complete
          ADMET and aqueous-solubility profile.
        </p>
      </div>

      <form
        className="smiles-form smiles-form-v2"
        onSubmit={handleSubmit}
      >
        <div className="smiles-form-glow" />

        <div className="smiles-workspace-header">
          <div className="workspace-heading">
            <div className="workspace-icon">
              <Binary size={24} />
            </div>

            <div>
              <span>Molecular input</span>
              <h3>SMILES representation</h3>
            </div>
          </div>

          <div className="input-status">
            <span />
            Ready for analysis
          </div>
        </div>

        <div className="input-pipeline">
          <div>
            <ScanLine size={17} />
            Parse structure
          </div>

          <span>→</span>

          <div>
            <ShieldCheck size={17} />
            Validate molecule
          </div>

          <span>→</span>

          <div>
            <Sparkles size={17} />
            Predict properties
          </div>
        </div>

        <div className="textarea-container">
          <textarea
            id="smiles-input"
            value={smiles}
            onChange={(event) =>
              setSmiles(event.target.value)
            }
            placeholder="Enter molecular SMILES, for example: CC(=O)OC1=CC=CC=C1C(=O)O"
            rows={5}
            spellCheck="false"
            disabled={loading}
          />

          <div className="textarea-corner-label">
            SMILES
          </div>

          <button
            type="button"
            className="clear-button clear-button-v2"
            onClick={() => setSmiles("")}
            disabled={loading}
          >
            <Eraser size={16} />
            Clear
          </button>
        </div>

        <div className="example-container example-container-v2">
          <div>
            <span className="example-title">
              Quick examples
            </span>

            <p>
              Select a known molecule to test the platform.
            </p>
          </div>

          <div className="example-buttons">
            {EXAMPLE_MOLECULES.map((molecule) => (
              <button
                type="button"
                key={molecule.id}
                onClick={() =>
                  setSmiles(molecule.smiles)
                }
                disabled={loading}
              >
                <span>{molecule.name}</span>
                <small>{molecule.formula}</small>
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="error-message">
            <span>!</span>
            {error}
          </div>
        )}

        <button
          type="submit"
          className="primary-button analyze-button analyze-button-v2"
          disabled={loading}
        >
          {loading ? (
            <>
              <span className="button-spinner" />
              Running molecular analysis...
            </>
          ) : (
            <>
              <Search size={21} />
              Analyze Molecular Profile
              <Sparkles size={19} />
            </>
          )}
        </button>

        <div className="input-footer-note">
          <ShieldCheck size={15} />

          The molecule is validated using RDKit before model
          inference.
        </div>
      </form>
    </section>
  );
}

export default SmilesInput;