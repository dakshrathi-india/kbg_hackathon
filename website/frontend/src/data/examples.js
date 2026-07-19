export const EXAMPLE_MOLECULES = [
  {
    id: "aspirin",
    name: "Aspirin",
    formula: "C9H8O4",
    description: "Analgesic and anti-inflammatory drug",
    smiles: "CC(=O)OC1=CC=CC=C1C(=O)O",
  },
  {
    id: "caffeine",
    name: "Caffeine",
    formula: "C8H10N4O2",
    description: "Central nervous system stimulant",
    smiles: "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
  },
  {
    id: "acetaminophen",
    name: "Acetaminophen",
    formula: "C8H9NO2",
    description: "Analgesic and antipyretic drug",
    smiles: "CC(=O)NC1=CC=C(C=C1)O",
  },
  {
    id: "ibuprofen",
    name: "Ibuprofen",
    formula: "C13H18O2",
    description: "Non-steroidal anti-inflammatory drug",
    smiles: "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
  },
];

export const DEFAULT_SMILES = EXAMPLE_MOLECULES[0].smiles;