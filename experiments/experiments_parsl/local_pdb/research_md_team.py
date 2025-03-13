# %%
import os
import sys

import dill
import parsl

from SimAgent.parsl_config import LocalConfig, WorkstationConfig
from SimAgent.teams import sim_team

# from parsl.configs.local_threads import config
os.environ["GMX_ff"] = "/homes/heng.ma/Research/force_field/charmm36-feb2021.ff"
os.environ["GMX_mdp"] = "/homes/heng.ma/Research/force_field/ions.mdp"

# %%
chain = sim_team(parsl_run=True)

# config = LocalConfig(max_workers_per_node=10).get_parsl_config(run_dir="./run_dir")
config = WorkstationConfig(available_accelerators=["2", "3", "4", "5"]).get_parsl_config(run_dir="./run_dir")

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
                "Can you run 8 simulations on local PDB file at /homes/heng.ma/Research/agent_lg/SimAgent/examples/experiments_parsl/local_pdb/2KKJ.pdb in 313 K for 50 ps? ",
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
