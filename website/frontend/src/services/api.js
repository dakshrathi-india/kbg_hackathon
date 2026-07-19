const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const USE_MOCK_API =
  import.meta.env.VITE_USE_MOCK_API !== "false";


const wait = (milliseconds) =>
  new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });


function createNumberGenerator(text) {
  let hash = 0;

  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }

  return function nextNumber() {
    hash = (hash * 1664525 + 1013904223) >>> 0;
    return hash / 4294967296;
  };
}


function round(value, decimalPlaces = 2) {
  const multiplier = 10 ** decimalPlaces;
  return Math.round(value * multiplier) / multiplier;
}


function probabilityLabel(
  probability,
  positiveLabel,
  negativeLabel,
) {
  return probability >= 0.5
    ? positiveLabel
    : negativeLabel;
}


function createMockMoleculeSvg() {
  return `
    <svg
      width="520"
      height="340"
      viewBox="0 0 520 340"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Molecular structure preview"
    >
      <defs>
        <linearGradient
          id="moleculeBackground"
          x1="0"
          y1="0"
          x2="1"
          y2="1"
        >
          <stop offset="0%" stop-color="#08111f" />
          <stop offset="100%" stop-color="#111827" />
        </linearGradient>

        <linearGradient
          id="bondGradient"
          x1="0"
          y1="0"
          x2="1"
          y2="1"
        >
          <stop offset="0%" stop-color="#22d3ee" />
          <stop offset="55%" stop-color="#3b82f6" />
          <stop offset="100%" stop-color="#8b5cf6" />
        </linearGradient>

        <filter id="moleculeGlow">
          <feGaussianBlur stdDeviation="4" result="blur" />

          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <rect
        width="520"
        height="340"
        rx="28"
        fill="url(#moleculeBackground)"
      />

      <g
        opacity="0.12"
        stroke="#67e8f9"
        stroke-width="1"
      >
        <line x1="0" y1="68" x2="520" y2="68" />
        <line x1="0" y1="136" x2="520" y2="136" />
        <line x1="0" y1="204" x2="520" y2="204" />
        <line x1="0" y1="272" x2="520" y2="272" />

        <line x1="104" y1="0" x2="104" y2="340" />
        <line x1="208" y1="0" x2="208" y2="340" />
        <line x1="312" y1="0" x2="312" y2="340" />
        <line x1="416" y1="0" x2="416" y2="340" />
      </g>

      <g
        stroke="url(#bondGradient)"
        stroke-width="8"
        stroke-linecap="round"
        stroke-linejoin="round"
        filter="url(#moleculeGlow)"
      >
        <line x1="135" y1="160" x2="195" y2="105" />
        <line x1="195" y1="105" x2="275" y2="125" />
        <line x1="275" y1="125" x2="320" y2="190" />
        <line x1="320" y1="190" x2="245" y2="235" />
        <line x1="245" y1="235" x2="165" y2="215" />
        <line x1="165" y1="215" x2="135" y2="160" />

        <line x1="275" y1="125" x2="365" y2="80" />
        <line x1="320" y1="190" x2="400" y2="235" />
        <line x1="135" y1="160" x2="75" y2="115" />
      </g>

      <g
        fill="#0f172a"
        stroke="#e2e8f0"
        stroke-width="5"
      >
        <circle cx="135" cy="160" r="23" />
        <circle cx="195" cy="105" r="23" />
        <circle cx="275" cy="125" r="23" />
        <circle cx="320" cy="190" r="23" />
        <circle cx="245" cy="235" r="23" />
        <circle cx="165" cy="215" r="23" />

        <circle
          cx="365"
          cy="80"
          r="25"
          stroke="#fb7185"
        />

        <circle
          cx="400"
          cy="235"
          r="25"
          stroke="#60a5fa"
        />

        <circle
          cx="75"
          cy="115"
          r="25"
          stroke="#fb7185"
        />
      </g>

      <g
        font-family="Arial, sans-serif"
        font-size="21"
        font-weight="800"
        text-anchor="middle"
        dominant-baseline="middle"
      >
        <text x="135" y="160" fill="#ffffff">C</text>
        <text x="195" y="105" fill="#ffffff">C</text>
        <text x="275" y="125" fill="#ffffff">C</text>
        <text x="320" y="190" fill="#ffffff">C</text>
        <text x="245" y="235" fill="#ffffff">C</text>
        <text x="165" y="215" fill="#ffffff">C</text>

        <text x="365" y="80" fill="#fb7185">O</text>
        <text x="400" y="235" fill="#60a5fa">N</text>
        <text x="75" y="115" fill="#fb7185">O</text>
      </g>

      <text
        x="260"
        y="310"
        text-anchor="middle"
        font-family="Arial, sans-serif"
        font-size="15"
        font-weight="600"
        fill="#94a3b8"
      >
        Molecular structure preview • RDKit rendering
      </text>
    </svg>
  `;
}


function createMockPrediction(smiles) {
  const random = createNumberGenerator(smiles);

  const distributionProbability = round(
    0.15 + random() * 0.8,
    3,
  );

  const metabolismProbability = round(
    0.1 + random() * 0.85,
    3,
  );

  const toxicityProbability = round(
    0.08 + random() * 0.86,
    3,
  );

  const absorptionValue = round(
    -6.5 + random() * 3,
    3,
  );

  const excretionValue = round(
    10 + random() * 85,
    3,
  );

  const solubilityValue = round(
    -6 + random() * 5,
    3,
  );

  const molecularWeight = round(
    120 + random() * 330,
    2,
  );

  const logP = round(
    -0.5 + random() * 5.5,
    2,
  );

  const tpsa = round(
    20 + random() * 130,
    2,
  );

  return {
    valid: true,

    original_smiles: smiles,

    canonical_smiles: smiles,

    molecule_svg: createMockMoleculeSvg(),

    descriptors: {
      molecular_weight: molecularWeight,
      logp: logP,
      tpsa,
      h_bond_donors: Math.floor(random() * 5),
      h_bond_acceptors: Math.floor(random() * 10),
      rotatable_bonds: Math.floor(random() * 12),
      ring_count: Math.floor(random() * 5),
      aromatic_ring_count: Math.floor(random() * 4),
      heavy_atom_count: Math.floor(
        8 + random() * 35,
      ),
      fraction_csp3: round(random(), 2),
    },

    predictions: {
      absorption: {
        code: "A",
        endpoint: "Absorption",
        property: "Caco-2 Permeability",
        type: "regression",
        value: absorptionValue,
        unit: "log permeability",
        label:
          absorptionValue > -5
            ? "Higher predicted permeability"
            : "Lower predicted permeability",
        model: "Final ensemble",
      },

      distribution: {
        code: "D",
        endpoint: "Distribution",
        property: "BBB Penetration",
        type: "classification",
        value: distributionProbability,
        unit: "probability",
        label: probabilityLabel(
          distributionProbability,
          "Likely BBB permeable",
          "Likely BBB non-permeable",
        ),
        model: "Final ensemble",
      },

      metabolism: {
        code: "M",
        endpoint: "Metabolism",
        property: "CYP3A4 Inhibition",
        type: "classification",
        value: metabolismProbability,
        unit: "probability",
        label: probabilityLabel(
          metabolismProbability,
          "Higher inhibition likelihood",
          "Lower inhibition likelihood",
        ),
        model: "Final ensemble",
      },

      excretion: {
        code: "E",
        endpoint: "Excretion",
        property: "Hepatocyte Clearance",
        type: "regression",
        value: excretionValue,
        unit: "dataset-specific clearance unit",
        label:
          excretionValue > 50
            ? "Higher predicted clearance"
            : "Lower predicted clearance",
        model: "Final ensemble",
      },

      toxicity: {
        code: "T",
        endpoint: "Toxicity",
        property: "AMES Mutagenicity",
        type: "classification",
        value: toxicityProbability,
        unit: "probability",
        label: probabilityLabel(
          toxicityProbability,
          "Higher mutagenicity likelihood",
          "Lower mutagenicity likelihood",
        ),
        model: "Final ensemble",
      },

      solubility: {
        code: "S",
        endpoint: "Solubility",
        property: "Aqueous Solubility",
        type: "regression",
        value: solubilityValue,
        unit: "log mol/L",
        label:
          solubilityValue > -3
            ? "Relatively higher aqueous solubility"
            : "Relatively lower aqueous solubility",
        model: "Final ensemble",
      },
    },

    metadata: {
      inference_mode: "mock",
      model_version: "frontend-development",
      disclaimer:
        "Mock output for frontend development only.",
    },
  };
}


async function predictUsingMockApi(smiles) {
  await wait(900);

  const cleanedSmiles = smiles.trim();

  if (!cleanedSmiles) {
    throw new Error("Please enter a SMILES string.");
  }

  if (cleanedSmiles.length < 2) {
    throw new Error(
      "The entered SMILES string is too short.",
    );
  }

  return createMockPrediction(cleanedSmiles);
}


async function predictUsingBackend(smiles) {
  const response = await fetch(
    `${API_BASE_URL}/api/predict`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        smiles: smiles.trim(),
      }),
    },
  );

  let responseBody;

  try {
    responseBody = await response.json();
  } catch {
    responseBody = null;
  }

  if (!response.ok) {
    const errorMessage =
      responseBody?.detail ||
      responseBody?.message ||
      "Prediction request failed.";

    throw new Error(errorMessage);
  }

  return responseBody;
}


export async function predictMolecule(smiles) {
  if (USE_MOCK_API) {
    return predictUsingMockApi(smiles);
  }

  return predictUsingBackend(smiles);
}