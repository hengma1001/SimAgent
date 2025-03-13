# %%
import os
import sys

import dill
import parsl

from SimAgent.parsl_config import LocalConfig, WorkstationConfig
from SimAgent.teams import research_sim_team

# from parsl.configs.local_threads import config
os.environ["GMX_ff"] = "/homes/heng.ma/Research/force_field/charmm36-feb2021.ff"
os.environ["GMX_mdp"] = "/homes/heng.ma/Research/force_field/ions.mdp"

# %%
chain = research_sim_team(parsl_run=True, n_search_results=10)

# config = LocalConfig(max_workers_per_node=10).get_parsl_config(run_dir="./run_dir")
config = WorkstationConfig(available_accelerators=8).get_parsl_config(run_dir="./run_dir")

if chain.parsl_run:
    print(config)
    parsl.load(config)

# sys.exit()

# %%
chain = chain.build_graph()

prompt = "Can you find and download 8 crystal structures of lysozyme from PDB, and run molecular dynamics simulation for each structure in 310 K for 100 ps? "

for chunk in chain.stream(
    {
        "messages": [
            (
                "human",
                prompt,
            )
        ]
    },
    stream_mode="values",
):
    print(chunk["messages"][-1])
    print("======================")
    with open("llm_md.log", "wb") as f:
        dill.dump(chunk, f)


parsl.dfk().cleanup()
