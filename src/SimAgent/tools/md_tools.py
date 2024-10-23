from pathlib import Path
from typing import Annotated, Optional

import openmm as omm
import parmed as pmd
from langchain_core.tools import tool
from md_setup.param import AMBER_param
from openmm import app
from openmm import unit as u


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


def build_top_tleap(pdb_file):
    amberParam = AMBER_param(pdb_file)
    amberParam.param_comp()
    return amberParam.output_top, amberParam.output_pdb
