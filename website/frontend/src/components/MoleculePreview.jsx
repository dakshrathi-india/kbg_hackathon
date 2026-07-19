import { useState } from "react";
import {
  Atom,
  Check,
  Clipboard,
} from "lucide-react";

function MoleculePreview({ result }) {
  const [copied, setCopied] = useState(false);

  const copyCanonicalSmiles = async () => {
    try {
      await navigator.clipboard.writeText(
        result.canonical_smiles,
      );

      setCopied(true);

      setTimeout(() => {
        setCopied(false);
      }, 1500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <section className="molecule-preview glass-card">
      <div className="result-section-header">
        <div>
          <div className="section-label">
            <Atom size={18} />
            Molecular Structure
          </div>

          <h3>Validated molecule</h3>
        </div>

        <div className="validation-badge">
          <Check size={16} />
          Valid SMILES
        </div>
      </div>

      <div className="molecule-preview-grid">
        <div
          className="molecule-svg"
          dangerouslySetInnerHTML={{
            __html: result.molecule_svg,
          }}
        />

        <div className="smiles-information">
          <div className="smiles-value">
            <span>Original SMILES</span>
            <p>{result.original_smiles}</p>
          </div>

          <div className="smiles-value">
            <div className="canonical-header">
              <span>Canonical SMILES</span>

              <button
                type="button"
                onClick={copyCanonicalSmiles}
                aria-label="Copy canonical SMILES"
              >
                {copied ? (
                  <Check size={17} />
                ) : (
                  <Clipboard size={17} />
                )}
              </button>
            </div>

            <p>{result.canonical_smiles}</p>
          </div>

          <div className="inference-note">
            <strong>Inference mode:</strong>{" "}
            {result.metadata?.inference_mode === "mock"
              ? "Frontend demonstration"
              : "Trained ML ensemble"}
          </div>
        </div>
      </div>
    </section>
  );
}

export default MoleculePreview;