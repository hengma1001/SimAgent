# %%
import os
import sys

import parsl

from SimAgent.parsl_config import LocalConfig, WorkstationConfig
from SimAgent.teams import research_sim_team

# from parsl.configs.local_threads import config
os.environ["GMX_ff"] = "/homes/heng.ma/Research/force_field/charmm36-feb2021.ff"
os.environ["GMX_mdp"] = "/homes/heng.ma/Research/force_field/ions.mdp"

# %%
chain = research_sim_team(parsl_run=True)

# config = LocalConfig(max_workers_per_node=10).get_parsl_config(run_dir="./run_dir")
config = WorkstationConfig(available_accelerators=["0", "7"]).get_parsl_config(run_dir="./run_dir")

if chain.parsl_run:
    print(config)
    parsl.load(config)

# sys.exit()

# %%
chain = chain.build_graph()


for chunk in chain.stream(
    {
        "messages": [
            (
                "human",
                "Can you find and download 5 lysozyme structures from PDB, and run simulations on them in 313 K for 50 ps? ",
            )
        ]
    },
    stream_mode="values",
):
    # chunk["messages"][-1].pretty_print()
    print(chunk["messages"][-1])
    print("======================")


parsl.dfk().cleanup()
