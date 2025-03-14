import os
import shutil
from pathlib import Path
from typing import Annotated, Optional

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pyemma
from langchain_core.tools import tool
from MDAnalysis.analysis import rms
from sklearn.neighbors import LocalOutlierFactor

from .md_tools import simulate_structure
from .utils import get_work_dir


# @tool
def run_enhance_md(
    pdb_files: Annotated[list, "a list of 3D structure in pdb format"],
    nonbondedCutoff: Annotated[float, "cutoff distance for nonbonded interactions"] = 1.0,
    hydrogenMass: Annotated[float, "mass of hydrogen atoms to stabilize protein dynamics"] = 1.0,
    pressure: Annotated[float, "pressure for NPT ensemble in atm"] = 1.0,
    temperature: Annotated[float, "simulation temperature in kelvin"] = 300,
    timestep: Annotated[float, "simulation timestep in ps"] = 0.002,
    report_frequency: Annotated[float, "How often MD writes a frame in ps"] = 10,
    simLength: Annotated[float, "The length of the simulation in ns"] = 0.1,
    num_sim: Annotated[int, "The number of simulations to run"] = 1,
    ref_pdb: Annotated[Optional[str], "Reference structure in pdb format for RMSD calculation"] = None,
    lag_frame: Annotated[int, "Lag time for TICA calculation"] = 5,
):
    """
    Run `num_sim` molecular dynamics simulation runs
    """
    run_insts = []
    for i in range(num_sim):
        pdb_file = pdb_files[i % len(pdb_files)]
        sim_inst = simulate_structure(
            pdb_file, nonbondedCutoff, hydrogenMass, pressure, temperature, timestep, report_frequency, simLength
        )
        run_insts.append(sim_inst)

    results = []
    for inst in run_insts:
        results.append(inst.result())

    results = find_outliers(results, ref_pdb=ref_pdb, lag_frame=lag_frame)
    return results


def find_outliers(
    results: Annotated[dict, "simulation results with pdb and dcd in dict"],
    ref_pdb: Annotated[Optional[str], "Reference structure in pdb format for RMSD calculation"] = None,
    lag_frame: Annotated[int, "Lag time for TICA calculation"] = 5,
):
    """
    Find the outliers in the trajectory files
    """
    work_dir = get_work_dir(tag="md_analysis")

    traj_files = []
    for i, result in enumerate(results):
        pdb_file = result["pdb_file"]
        traj_file = result["trajectory"]

        label = os.path.basename(os.path.dirname(traj_file))

        mda_u = mda.Universe(pdb_file, traj_file)
        protein = mda_u.select_atoms("protein")
        if i == 0:
            pdb_nosol = f"{work_dir}/{os.path.basename(pdb_file)}"
            protein.write(pdb_nosol)

        dcd_file = f"{work_dir}/{label}.dcd"
        with mda.Writer(dcd_file, protein.n_atoms) as w:
            for ts in mda_u.trajectory:
                w.write(protein)
        traj_files.append(dcd_file)

    torsions_feat = pyemma.coordinates.featurizer(pdb_nosol)
    torsions_feat.add_backbone_torsions(cossin=True, periodic=False)
    torsions_data = pyemma.coordinates.load(traj_files, features=torsions_feat)
    tica = pyemma.coordinates.tica(torsions_data, lag=lag_frame)
    tica_output = tica.get_output()
    tica_concatenated = np.concatenate(tica_output)

    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred = clf.fit_predict(tica_concatenated)
    # n_errors = (y_pred != ground_truth).sum()
    X_scores = clf.negative_outlier_factor_

    mda_u = mda.Universe(pdb_nosol, traj_files)
    protein = mda_u.select_atoms("protein")
    if ref_pdb:
        ref_u = mda.Universe(ref_pdb)
    else:
        ref_u = mda_u
    rmsd = rms.RMSD(mda_u, ref_u, select="protein and name CA", ref_frame=0)
    rmsd.run()
    rmsd_output = rmsd.rmsd.T

    df = pd.DataFrame(
        {
            "frame": np.arange(len(X_scores), dtype=int),
            "rmsd": rmsd_output[2],
            "lof": X_scores,
        }
    )

    df.sort_values("lof", inplace=True)
    df.sort_values("rmsd", inplace=True)
    df.to_pickle(f"{work_dir}/result.pkl")

    top_outliers = df.head(10)

    pdb_files = []
    for i, row in top_outliers.iterrows():
        mda_u.trajectory[int(row["frame"])]
        pdb_file = f"{work_dir}/outliers_{i}.pdb"
        protein.write(pdb_file)
        pdb_files.append(pdb_file)

    if df["rmsd"].min() < 1:
        return f"protein folded, no outliers found"
    else:
        return f"Found outliers from simulation, stored at {pdb_files}, lowest RMSD is {df['rmsd'].min()}"
