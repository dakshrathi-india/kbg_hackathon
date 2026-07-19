from typing import Callable

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Lipinski,
    rdDepictor,
    rdMolDescriptors,
)

from rdkit.Chem.Draw import rdMolDraw2D


MORGAN_RADIUS = 2
MORGAN_BITS = 2048


MODEL_DESCRIPTOR_FUNCTIONS: list[
    tuple[str, Callable]
] = [
    (
        "molecular_weight",
        Descriptors.MolWt,
    ),
    (
        "logp",
        Descriptors.MolLogP,
    ),
    (
        "tpsa",
        rdMolDescriptors.CalcTPSA,
    ),
    (
        "h_bond_donors",
        Lipinski.NumHDonors,
    ),
    (
        "h_bond_acceptors",
        Lipinski.NumHAcceptors,
    ),
    (
        "rotatable_bonds",
        Lipinski.NumRotatableBonds,
    ),
    (
        "ring_count",
        rdMolDescriptors.CalcNumRings,
    ),
    (
        "aromatic_ring_count",
        rdMolDescriptors.CalcNumAromaticRings,
    ),
    (
        "heavy_atom_count",
        Descriptors.HeavyAtomCount,
    ),
    (
        "fraction_csp3",
        rdMolDescriptors.CalcFractionCSP3,
    ),
    (
        "heteroatom_count",
        rdMolDescriptors.CalcNumHeteroatoms,
    ),
    (
        "molar_refractivity",
        Descriptors.MolMR,
    ),
]


def parse_smiles(
    smiles: str,
) -> tuple[Chem.Mol, str]:
    cleaned_smiles = smiles.strip()

    if not cleaned_smiles:
        raise ValueError(
            "Please enter a SMILES string."
        )

    molecule = Chem.MolFromSmiles(
        cleaned_smiles
    )

    if molecule is None:
        raise ValueError(
            "Invalid SMILES string. RDKit could not "
            "construct a valid molecule."
        )

    canonical_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )

    return molecule, canonical_smiles


def build_display_descriptors(
    molecule: Chem.Mol,
) -> dict[str, float | int]:
    values = {}

    for name, function in MODEL_DESCRIPTOR_FUNCTIONS:
        value = function(molecule)

        if isinstance(value, float):
            values[name] = round(
                float(value),
                4,
            )
        else:
            values[name] = int(value)

    return values


def build_model_features(
    molecule: Chem.Mol,
) -> np.ndarray:
    fingerprint = (
        AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            radius=MORGAN_RADIUS,
            nBits=MORGAN_BITS,
            useChirality=True,
        )
    )

    fingerprint_array = np.zeros(
        MORGAN_BITS,
        dtype=np.float32,
    )

    DataStructs.ConvertToNumpyArray(
        fingerprint,
        fingerprint_array,
    )

    descriptor_array = np.asarray(
        [
            float(function(molecule))
            for _, function
            in MODEL_DESCRIPTOR_FUNCTIONS
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            fingerprint_array,
            descriptor_array,
        ]
    )


def render_molecule_svg(
    molecule: Chem.Mol,
) -> str:
    drawable_molecule = Chem.Mol(
        molecule
    )

    rdDepictor.Compute2DCoords(
        drawable_molecule
    )

    drawer = rdMolDraw2D.MolDraw2DSVG(
        520,
        340,
    )

    options = drawer.drawOptions()

    options.clearBackground = False
    options.bondLineWidth = 2.2
    options.padding = 0.08

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        drawable_molecule,
    )

    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()

    return svg.replace(
        "svg:",
        "",
    )