import {
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  Trophy,
} from "lucide-react";

import { MODEL_RESULTS } from "../data/modelResults";

function formatScore(score) {
  if (score === null || score === undefined) {
    return "Pending";
  }

  return Number(score).toFixed(4);
}

function PerformanceSection() {
  return (
    <section
      id="performance"
      className="performance-section page-section"
    >
      <div className="section-heading">
        <div className="section-label">
          <BarChart3 size={18} />
          Model Transparency
        </div>

        <h2>Endpoint-specific model performance</h2>

        <p>
          Every endpoint is independently trained and
          evaluated using molecular scaffold-aware splits.
        </p>
      </div>

      <div className="performance-grid">
        {MODEL_RESULTS.map((result) => (
          <article
            className="performance-card glass-card"
            key={result.code}
          >
            <div className="performance-card-header">
              <div className="performance-code">
                {result.code}
              </div>

              <div
                className={`status-badge ${
                  result.status === "completed"
                    ? "status-completed"
                    : "status-training"
                }`}
              >
                {result.status === "completed" ? (
                  <CheckCircle2 size={14} />
                ) : (
                  <Clock3 size={14} />
                )}

                {result.status}
              </div>
            </div>

            <span className="performance-endpoint">
              {result.endpoint}
            </span>

            <h3>{result.property}</h3>

            <div className="dataset-name">
              <Database size={15} />
              {result.dataset}
            </div>

            <div className="selected-model">
              <Trophy size={17} />

              <div>
                <span>Selected solution</span>
                <strong>{result.selectedModel}</strong>
              </div>
            </div>

            <div className="metric-grid">
              <div>
                <span>{result.primaryMetric}</span>
                <strong>
                  {formatScore(result.primaryScore)}
                </strong>
              </div>

              <div>
                <span>{result.secondaryMetric}</span>
                <strong>
                  {formatScore(result.secondaryScore)}
                </strong>
              </div>
            </div>

            <div className="task-type">
              {result.taskType}
            </div>
          </article>
        ))}
      </div>

      <div className="evaluation-note glass-card">
        <strong>Evaluation policy:</strong> models are
        selected using validation data. The final test set
        remains untouched during model and ensemble
        selection.
      </div>
    </section>
  );
}

export default PerformanceSection;