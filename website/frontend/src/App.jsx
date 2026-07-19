import { useState } from "react";

import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import SmilesInput from "./components/SmilesInput";
import LoadingState from "./components/LoadingState";
import MoleculePreview from "./components/MoleculePreview";
import PredictionGrid from "./components/PredictionGrid";
import DescriptorPanel from "./components/DescriptorPanel";
import PerformanceSection from "./components/PerformanceSection";
import MethodologySection from "./components/MethodologySection";
import Footer from "./components/Footer";

import { DEFAULT_SMILES } from "./data/examples";
import { predictMolecule } from "./services/api";

function App() {
  const [smiles, setSmiles] = useState(DEFAULT_SMILES);
  const [predictionResult, setPredictionResult] =
    useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeMolecule = async () => {
    const cleanedSmiles = smiles.trim();

    if (!cleanedSmiles) {
      setError("Please enter a molecular SMILES string.");
      return;
    }

    setLoading(true);
    setError("");
    setPredictionResult(null);

    try {
      const result = await predictMolecule(cleanedSmiles);

      setPredictionResult(result);

      setTimeout(() => {
        document
          .getElementById("prediction-results")
          ?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
      }, 100);
    } catch (requestError) {
      setError(
        requestError.message ||
          "Unable to analyze this molecule.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Navbar />

      <main>
        <Hero />

        <SmilesInput
          smiles={smiles}
          setSmiles={setSmiles}
          onAnalyze={analyzeMolecule}
          loading={loading}
          error={error}
        />

        {loading && <LoadingState />}

        {predictionResult && (
          <div
            id="prediction-results"
            className="results-container page-section"
          >
            <MoleculePreview result={predictionResult} />

            <PredictionGrid
              predictions={
                predictionResult.predictions
              }
            />

            <DescriptorPanel
              descriptors={
                predictionResult.descriptors
              }
            />
          </div>
        )}

        <PerformanceSection />
        <MethodologySection />
      </main>

      <Footer />
    </div>
  );
}

export default App;