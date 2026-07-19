import {
  Boxes,
  CircleDot,
  Droplets,
  Gauge,
  Hexagon,
  Layers3,
  Orbit,
  RotateCw,
  Scale,
  TestTube2,
} from "lucide-react";

const DESCRIPTOR_CONFIG = [
  {
    key: "molecular_weight",
    label: "Molecular Weight",
    unit: "g/mol",
    icon: Scale,
  },
  {
    key: "logp",
    label: "LogP",
    unit: "",
    icon: Droplets,
  },
  {
    key: "tpsa",
    label: "TPSA",
    unit: "Å²",
    icon: Gauge,
  },
  {
    key: "h_bond_donors",
    label: "H-bond Donors",
    unit: "",
    icon: TestTube2,
  },
  {
    key: "h_bond_acceptors",
    label: "H-bond Acceptors",
    unit: "",
    icon: CircleDot,
  },
  {
    key: "rotatable_bonds",
    label: "Rotatable Bonds",
    unit: "",
    icon: RotateCw,
  },
  {
    key: "ring_count",
    label: "Ring Count",
    unit: "",
    icon: Hexagon,
  },
  {
    key: "aromatic_ring_count",
    label: "Aromatic Rings",
    unit: "",
    icon: Orbit,
  },
  {
    key: "heavy_atom_count",
    label: "Heavy Atoms",
    unit: "",
    icon: Boxes,
  },
  {
    key: "fraction_csp3",
    label: "Fraction Csp3",
    unit: "",
    icon: Layers3,
  },
];

function DescriptorPanel({ descriptors }) {
  return (
    <section className="descriptor-section">
      <div className="result-section-header">
        <div>
          <div className="section-label">
            Molecular Descriptors
          </div>

          <h3>Interpretable chemical features</h3>
        </div>
      </div>

      <div className="descriptor-grid">
        {DESCRIPTOR_CONFIG.map(
          ({ key, label, unit, icon: Icon }) => (
            <article
              className="descriptor-card glass-card"
              key={key}
            >
              <div className="descriptor-icon">
                <Icon size={20} />
              </div>

              <div>
                <span>{label}</span>

                <strong>
                  {descriptors[key] ?? "—"}
                  {unit && (
                    <small>
                      {" "}
                      {unit}
                    </small>
                  )}
                </strong>
              </div>
            </article>
          ),
        )}
      </div>
    </section>
  );
}

export default DescriptorPanel;