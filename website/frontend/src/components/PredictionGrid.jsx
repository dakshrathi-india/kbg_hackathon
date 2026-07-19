import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import PredictionCard from "./PredictionCard";

function PredictionGrid({ predictions }) {
  const predictionList = Object.values(predictions);

  const radarData = [
    {
      endpoint: "BBB",
      score: Math.round(
        predictions.distribution.value * 100,
      ),
    },
    {
      endpoint: "CYP3A4",
      score: Math.round(
        predictions.metabolism.value * 100,
      ),
    },
    {
      endpoint: "AMES",
      score: Math.round(
        predictions.toxicity.value * 100,
      ),
    },
  ];

  return (
    <section className="predictions-section">
      <div className="result-section-header">
        <div>
          <div className="section-label">
            AI Prediction Profile
          </div>

          <h3>ADMET and developability results</h3>
        </div>

        <span className="six-properties-badge">
          6 predicted properties
        </span>
      </div>

      <div className="prediction-layout">
        <div className="prediction-grid">
          {predictionList.map((prediction) => (
            <PredictionCard
              key={prediction.code}
              prediction={prediction}
            />
          ))}
        </div>

        <div className="radar-card glass-card">
          <div>
            <span className="radar-eyebrow">
              Classification profile
            </span>

            <h4>Model-estimated likelihoods</h4>

            <p>
              Relative probability scores for BBB
              penetration, CYP3A4 inhibition and AMES
              mutagenicity.
            </p>
          </div>

          <div className="radar-wrapper">
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(148,163,184,0.25)" />

                <PolarAngleAxis
                  dataKey="endpoint"
                  tick={{
                    fill: "#cbd5e1",
                    fontSize: 13,
                  }}
                />

                <Tooltip
                  contentStyle={{
                    background: "#101827",
                    border:
                      "1px solid rgba(34,211,238,0.3)",
                    borderRadius: "12px",
                  }}
                  formatter={(value) => [
                    `${value}%`,
                    "Predicted score",
                  ]}
                />

                <Radar
                  name="Prediction score"
                  dataKey="score"
                  stroke="#22d3ee"
                  fill="#22d3ee"
                  fillOpacity={0.24}
                  strokeWidth={3}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}

export default PredictionGrid;