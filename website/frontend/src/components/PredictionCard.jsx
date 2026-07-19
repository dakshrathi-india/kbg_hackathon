import {
  Activity,
  Brain,
  Dna,
  Droplets,
  ShieldAlert,
  Wind,
} from "lucide-react";

const ICONS = {
  A: Activity,
  D: Brain,
  M: Dna,
  E: Wind,
  T: ShieldAlert,
  S: Droplets,
};

function formatPrediction(prediction) {
  if (prediction.type === "classification") {
    return `${Math.round(prediction.value * 100)}%`;
  }

  return Number(prediction.value).toFixed(3);
}

function getInterpretationTone(prediction) {
  const value = Number(prediction.value);

  switch (prediction.code) {
    case "A":
      return value > -5 ? "positive" : "negative";

    case "M":
      return value >= 0.5 ? "negative" : "positive";

    case "T":
      return value >= 0.5 ? "negative" : "positive";

    case "S":
      return value > -3 ? "positive" : "negative";

    case "D":
      return "informational";

    case "E":
      return "neutral";

    default:
      return "neutral";
  }
}

function PredictionCard({ prediction }) {
  const Icon = ICONS[prediction.code] || Activity;

  const probability =
    prediction.type === "classification"
      ? Math.max(
          0,
          Math.min(100, prediction.value * 100),
        )
      : null;

  const interpretationTone =
    getInterpretationTone(prediction);

  return (
    <article
      className={`prediction-card endpoint-${prediction.code.toLowerCase()}`}
    >
      <div className="prediction-card-top">
        <div className="prediction-icon">
          <Icon size={23} />
        </div>

        <span className="endpoint-code">
          {prediction.code}
        </span>
      </div>

      <span className="prediction-endpoint">
        {prediction.endpoint}
      </span>

      <h4>{prediction.property}</h4>

      <div className="prediction-value">
        {formatPrediction(prediction)}
      </div>

      <span className="prediction-unit">
        {prediction.unit}
      </span>

      {probability !== null && (
        <div className="probability-track">
          <div
            className="probability-fill"
            style={{
              width: `${probability}%`,
            }}
          />
        </div>
      )}

      <div
        className={`prediction-interpretation interpretation-${interpretationTone}`}
      >
        <span />

        {prediction.label}
      </div>

      <div className="prediction-model">
        {prediction.model}
      </div>
    </article>
  );
}

export default PredictionCard;