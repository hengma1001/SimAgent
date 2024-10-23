from pathlib import Path
from typing import Annotated, Optional

import openmm as omm
import parmed as pmd
import torch
from langchain_core.tools import tool
from openmm import app
from openmm import unit as u

from .registry import clear_torch_cuda_memory_callback, register


@tool
def simulate_structure(
    pdb_file: Annotated[str, "3D structure in pdb format"],
):
    """
    Model the molecular structure with molecular dynamics simulation

    Parameters
    ----------
    pdb_file : str
        3D structure in pdb format
    """
    top_file, pdb_file = build_top_tleap(pdb_file)
    top = pmd.load_file(top_file, xyz=pdb_file)

    app.PDBFile.writeFile(top.topology, top.positions, open("input.pdb", "w"))
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * u.nanometer,
        constraints=app.HBonds,
        hydrogenMass=4 * u.amu,
    )
    barostat = omm.MonteCarloBarostat(1.0 * u.atmosphere, 300 * u.kelvin)
    system.addForce(barostat)
    save_omm_system(system, "system.xml")

    integrator = omm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 0.004 * u.picoseconds)
    simulation = app.Simulation(top.topology, system, integrator)

    simulation.context.setPositions(top.positions)

    simulation.minimizeEnergy()
    simulation.reporters.append(app.DCDReporter("output.dcd", 1000))
    simulation.reporters.append(
        app.StateDataReporter(
            "output.log", 1000, step=True, time=True, potentialEnergy=True, temperature=True, speed=True
        )
    )
    simulation.step(10000)


def save_omm_system(system, save_xml):
    # Serialize the system to an xml file. This is will be necessary for resuming a simulation
    with open(save_xml, "w") as f:
        f.write(omm.XmlSerializer.serialize(system))


def load_omm_system(system, save_xml):
    # Load system
    with open(save_xml, "r") as f:
        system = omm.XmlSerializer.deserialize(f.read())
    return system


@tool
def fold_sequence(
    sequence: Annotated[str, "Protein amino acid sequence"],
    output_pdb: Annotated[str, "Path to save the generated protein structure"],
):
    """
    Tool to fold protein sequence into 3D structure.
    """
    esmfold = EsmFold()
    ppi_structure = esmfold.run(sequence)
    with open(output_pdb, "w") as f:
        f.write(ppi_structure)


@register(shutdown_callback=clear_torch_cuda_memory_callback)
class EsmFold:
    """ESM-Fold model for protein structure prediction."""

    def __init__(self, use_gpu: bool = True, torch_hub_dir: Optional[str] = None):
        # Configure the torch hub directory
        if torch_hub_dir is not None:
            torch.hub.set_dir(torch_hub_dir)

        # Load the model
        self.model = torch.hub.load("facebookresearch/esm:main", "esmfold_v1")
        if use_gpu:
            self.model.eval().cuda()

        # Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
        # Lower sizes will have lower memory requirements at the cost of increased speed.
        self.model.set_chunk_size(128)

    def run(self, sequence: str):
        """Run the ESMFold model to predict structure.

        Parameters
        ----------
        sequence : str
            The sequence to fold.
        """
        with torch.no_grad():
            return self.model.infer_pdb(sequence)
