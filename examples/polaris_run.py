import os
import sys

import dill
import parsl

from SimAgent.parsl_config import PolarisConfig
from SimAgent.teams import sim_team

current_dir = os.getcwd()

chain = sim_team()

# config = LocalConfig(max_workers_per_node=10).get_parsl_config(run_dir="./run_dir")
# config = WorkstationConfig(available_accelerators=["0", "7"]).get_parsl_config(run_dir="./run_dir")

polaris_internet = "source /lus/eagle/projects/FoundEpidem/hengma/internet.sh"

config = PolarisConfig(
    num_nodes=25,
    account="FoundEpidem",
    queue="prod",
    walltime="15:00",
    scheduler_options="#PBS -l filesystems=home:eagle",
    worker_init=f"source activate /lus/eagle/projects/FoundEpidem/hengma/conda_envs/simagent;export GMX_mdp=/lus/eagle/projects/FoundEpidem/hengma/sim_agent/experiments/data/ions.mdp;export GMX_ff=/lus/eagle/projects/FoundEpidem/hengma/sim_agent/experiments/data/charmm36-feb2021.ff;{polaris_internet};cd {current_dir}",
).get_parsl_config(run_dir="./run_dir")


print(config)
parsl.load(config)

chain = chain.build_graph()

prompt = (
    "Can you download the structure of 2KKJ from Protein Data Bank, and run 100 simulations of it in 313 K for 50 ps?"
)

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
    # chunk["messages"][-1].pretty_print()
    print(chunk["messages"][-1])
    print("======================")
    with open("llm_md.log", "wb") as f:
        dill.dump(chunk, f)


parsl.dfk().cleanup()

