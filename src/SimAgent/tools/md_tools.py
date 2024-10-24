import os
import urllib.request
from pathlib import Path
from typing import Annotated, Optional

import MDAnalysis as mda
import openmm as omm
import parmed as pmd
from langchain_core.tools import tool
from md_setup.param import AMBER_param
from openmm import app
from openmm import unit as u


@tool
def simulate_structure(
    pdb_file: Annotated[str, "3D structure in pdb format"],
    nonbondedCutoff: Annotated[float, "cutoff distance for nonbonded interactions"] = 1.0,
    hydrogenMass: Annotated[float, "mass of hydrogen atoms to stabilize protein dynamics"] = 1.0,
    pressure: Annotated[float, "pressure for NPT ensemble in atm"] = 1.0,
    temperature: Annotated[float, "simulation temperature in kelvin"] = 300,
    timestep: Annotated[float, "simulation timestep in ps"] = 0.002,
    simLength: Annotated[float, "The length of the simulation"] = 0.1,
):
    """
    Model the molecular structure with molecular dynamics simulation
    """
    top_file, pdb_file = build_top_tleap(pdb_file)
    top = pmd.load_file(top_file, xyz=pdb_file)

    app.PDBFile.writeFile(top.topology, top.positions, open("input.pdb", "w"))
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=nonbondedCutoff * u.nanometer,
        constraints=app.HBonds,
        hydrogenMass=hydrogenMass * u.amu,
    )
    barostat = omm.MonteCarloBarostat(pressure * u.atmosphere, temperature * u.kelvin)
    system.addForce(barostat)
    save_omm_system(system, "system.xml")

    integrator = omm.LangevinMiddleIntegrator(temperature * u.kelvin, 1 / u.picosecond, timestep * u.picoseconds)
    simulation = app.Simulation(top.topology, system, integrator)

    simulation.context.setPositions(top.positions)

    simulation.minimizeEnergy()
    simulation.reporters.append(app.DCDReporter("output.dcd", 1000))
    simulation.reporters.append(
        app.StateDataReporter(
            "output.log", 1000, step=True, time=True, potentialEnergy=True, temperature=True, speed=True
        )
    )
    total_steps = (simLength * u.nanoseconds) / (timestep * u.picoseconds)
    simulation.step(total_steps)


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
    """
    Building protein topology with tleap,
    NOTE: assuming protein-only system
    """
    pdb_file = pick_protein_only(pdb_file)
    amberParam = AMBER_param(
        pdb_file,
        forcefield="ff14SB",
        watermodel="tip3p",
    )
    amberParam.param_comp()
    return amberParam.output_top, amberParam.output_inpcrd


def pick_protein_only(pdb_file):
    save_pdb = f"./_{os.path.basename(pdb_file)}"
    mda_u = mda.Universe(pdb_file)
    protein = mda_u.select_atoms("protein")
    protein.write(save_pdb)
    return save_pdb


@tool
def download_structure(pdb_code: Annotated[str, "PDB code for the protein"]):
    """
    Download the PDB structure from RCSB
    """
    url = f"https://files.rcsb.org/view/{pdb_code}.pdb"
    try:
        urllib.request.urlretrieve(url, f"{pdb_code}.pdb")
        return f"Successfully retrieved {pdb_code}.pdb."
    except:
        return f"Failed to retrieve {pdb_code}. Please recheck the PDB ID."
