"""Utilities to build Parsl configurations."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Sequence, Union

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self

from parsl.addresses import address_by_interface
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher, WrappedLauncher
from parsl.providers import LocalProvider, PBSProProvider
from pydantic import BaseModel, Field, model_validator

PathLike = Union[str, Path]


class BaseComputeConfig(BaseModel, ABC):
    """Compute config (HPC platform, number of GPUs, etc)."""

    # Name of the platform to uniquely identify it
    name: Literal[""] = ""

    @abstractmethod
    def get_parsl_config(self, run_dir: Union[str, Path]) -> Config:
        """Create a new Parsl configuration.

        Parameters
        ----------
        run_dir : str | Path
            Path to store monitoring DB and parsl logs.

        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class LocalConfig(BaseComputeConfig):
    """Local compute config."""

    name: Literal["local"] = "local"  # type: ignore[assignment]

    max_workers_per_node: int = Field(
        default=1,
        description="Number of workers to use.",
    )
    cores_per_worker: float = Field(
        default=1.0,
        description="Number of cores per worker.",
    )
    worker_port_range: tuple[int, int] = Field(
        default=(10000, 20000),
        description="Port range for the workers.",
    )
    label: str = Field(
        default="cpu_htex",
        description="Label for the executor.",
    )

    def get_parsl_config(self, run_dir: Union[str, Path]) -> Config:
        """Generate a Parsl configuration for local execution."""
        return Config(
            run_dir=str(run_dir),
            strategy=None,
            executors=[
                HighThroughputExecutor(
                    address="localhost",
                    label=self.label,
                    max_workers_per_node=self.max_workers_per_node,
                    cores_per_worker=self.cores_per_worker,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class WorkstationConfig(BaseComputeConfig):
    """Compute config for a workstation."""

    name: Literal["workstation"] = "workstation"  # type: ignore[assignment]

    available_accelerators: Union[int, Sequence[str]] = Field(
        default=1,
        description="Number of GPU accelerators to use.",
    )
    worker_port_range: tuple[int, int] = Field(
        default=(10000, 20000),
        description="Port range for the workers.",
    )
    retries: int = Field(
        default=1,
        description="Number of retries for the task.",
    )
    label: str = Field(
        default="gpu_htex",
        description="Label for the executor.",
    )
    worker_init: str = Field("", description="Command to run before starting a worker.")

    def get_parsl_config(self, run_dir: Union[str, Path]) -> Config:
        """Generate a Parsl configuration for workstation execution."""
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    address="localhost",
                    label=self.label,
                    cpu_affinity="block",
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(worker_init=self.worker_init, init_blocks=1, max_blocks=1),
                ),
            ],
        )


class WorkstationV2Config(BaseComputeConfig):
    """Compute config for a workstation."""

    name: Literal["workstation_v2"] = "workstation_v2"  # type: ignore[assignment]

    available_accelerators: Union[int, Sequence[str]] = Field(
        description="Number of GPU accelerators to use.",
    )
    worker_port_range: tuple[int, int] = Field(
        default=(10000, 20000),
        description="Port range for the workers.",
    )
    retries: int = Field(
        default=1,
        description="Number of retries for the task.",
    )
    # We have a long idletime to ensure train/inference executors are not
    # shut down (to enable warmstarts) while simulations are running.
    max_idletime: float = Field(
        default=60.0 * 10,
        description="The maximum idle time allowed for an executor before "
        "strategy could shut down unused blocks. Default is 10 minutes.",
    )

    @model_validator(mode="after")
    def validate_available_accelerators(self) -> Self:
        """Check there are at least 3 GPUs."""
        min_gpus = 3
        gpus = self.available_accelerators
        num_gpus = gpus if isinstance(gpus, int) else len(gpus)
        if num_gpus < min_gpus:
            raise ValueError("Must use at least 3 GPUs.")

        return self

    def _get_htex(
        self,
        label: str,
        available_accelerators: Sequence[str],
    ) -> HighThroughputExecutor:
        return HighThroughputExecutor(
            address="localhost",
            label=label,
            cpu_affinity="block",
            available_accelerators=available_accelerators,
            worker_port_range=self.worker_port_range,
            provider=LocalProvider(init_blocks=1, max_blocks=1),
        )

    def get_parsl_config(self, run_dir: Union[str, Path]) -> Config:
        """Generate a Parsl configuration."""
        # Handle the case where available_accelerators is an int
        accelerators = self.available_accelerators
        if isinstance(accelerators, int):
            accelerators = [str(i) for i in range(accelerators)]

        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            max_idletime=self.max_idletime,
            executors=[
                # Assign 1 GPU each for training and inference
                self._get_htex("train_htex", accelerators[:1]),
                self._get_htex("inference_htex", accelerators[1:2]),
                # Assign the remaining GPUs to simulation
                self._get_htex("simulation_htex", accelerators[2:]),
            ],
        )


class HybridWorkstationConfig(BaseComputeConfig):
    """Run simulations on CPU and AI models on GPU."""

    name: Literal["hybrid_workstation"] = "hybrid_workstation"  # type: ignore[assignment]

    cpu_config: LocalConfig = Field(
        description="Config for the CPU executor to run simulations.",
    )
    gpu_config: WorkstationConfig = Field(
        description="Config for the GPU executor to run AI models.",
    )

    def get_parsl_config(self, run_dir: Union[str, Path]) -> Config:
        """Generate a Parsl configuration for hybrid execution."""
        return Config(
            run_dir=str(run_dir),
            retries=self.gpu_config.retries,
            executors=[
                HighThroughputExecutor(
                    address="localhost",
                    label=self.cpu_config.label,
                    max_workers_per_node=self.cpu_config.max_workers_per_node,
                    cores_per_worker=self.cpu_config.cores_per_worker,
                    worker_port_range=self.cpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
                HighThroughputExecutor(
                    address="localhost",
                    label=self.gpu_config.label,
                    cpu_affinity="block",
                    available_accelerators=self.gpu_config.available_accelerators,
                    worker_port_range=self.gpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class InferenceTrainWorkstationConfig(BaseComputeConfig):
    """Run simulations on CPU and AI models on GPU."""

    name: Literal["inference_train_workstation"] = "inference_train_workstation"  # type: ignore[assignment]

    cpu_config: LocalConfig = Field(
        description="Config for the CPU executor to run simulations.",
    )
    train_gpu_config: WorkstationConfig = Field(
        description="Config for the GPU executor to run AI models.",
    )
    inference_gpu_config: WorkstationConfig = Field(
        description="Config for the GPU executor to run AI models.",
    )

    # We have a long idletime to ensure train/inference executors are not
    # shut down (to enable warmstarts) while simulations are running.
    max_idletime: float = Field(
        default=60.0 * 10,
        description="The maximum idle time allowed for an executor before "
        "strategy could shut down unused blocks. Default is 10 minutes.",
    )

    @model_validator(mode="after")
    def validate_htex_labels(self) -> Self:
        """Ensure that the labels are unique."""
        self.cpu_config.label = "simulation_htex"
        self.train_gpu_config.label = "train_htex"
        self.inference_gpu_config.label = "inference_htex"
        return self

    def get_parsl_config(self, run_dir: Union[str, Path]) -> Config:
        """Generate a Parsl configuration for hybrid execution."""
        return Config(
            run_dir=str(run_dir),
            retries=self.train_gpu_config.retries,
            max_idletime=self.max_idletime,
            executors=[
                HighThroughputExecutor(
                    address="localhost",
                    label=self.cpu_config.label,
                    max_workers_per_node=self.cpu_config.max_workers_per_node,
                    cores_per_worker=self.cpu_config.cores_per_worker,
                    worker_port_range=self.cpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
                HighThroughputExecutor(
                    address="localhost",
                    label=self.train_gpu_config.label,
                    cpu_affinity="block",
                    available_accelerators=self.train_gpu_config.available_accelerators,
                    worker_port_range=self.train_gpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
                HighThroughputExecutor(
                    address="localhost",
                    label=self.inference_gpu_config.label,
                    cpu_affinity="block",
                    available_accelerators=self.inference_gpu_config.available_accelerators,
                    worker_port_range=self.inference_gpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class VistaConfig(BaseComputeConfig):
    """VISTA compute config.

    https://tacc.utexas.edu/systems/vista/
    """

    name: Literal["vista"] = "vista"  # type: ignore[assignment]

    num_nodes: int = Field(
        ge=3,
        description="Number of nodes to use (must use at least 3 nodes).",
    )

    # We have a long idletime to ensure train/inference executors are not
    # shut down (to enable warmstarts) while simulations are running.
    max_idletime: float = Field(
        default=60.0 * 10,
        description="The maximum idle time allowed for an executor before "
        "strategy could shut down unused blocks. Default is 10 minutes.",
    )

    def _get_htex(self, label: str, num_nodes: int) -> HighThroughputExecutor:
        return HighThroughputExecutor(
            label=label,
            available_accelerators=1,  # 1 GH per node
            cores_per_worker=72,
            cpu_affinity="alternating",
            prefetch_capacity=0,
            provider=LocalProvider(
                launcher=WrappedLauncher(
                    prepend=f"srun -l --ntasks-per-node=1 --nodes={num_nodes}",
                ),
                cmd_timeout=120,
                nodes_per_block=num_nodes,
                init_blocks=1,
                max_blocks=1,
            ),
        )

    def get_parsl_config(self, run_dir: Union[str, Path]) -> Config:
        """Generate a Parsl configuration."""
        return Config(
            run_dir=str(run_dir),
            max_idletime=self.max_idletime,
            executors=[
                # Assign 1 node each for training and inference
                self._get_htex("train_htex", 1),
                self._get_htex("inference_htex", 1),
                # Assign the remaining nodes to simulation
                self._get_htex("simulation_htex", self.num_nodes - 2),
            ],
        )


class PolarisConfig(BaseComputeConfig):
    """Polaris@ALCF settings.

    See here for details: https://docs.alcf.anl.gov/polaris/workflows/parsl/
    """

    name: Literal["polaris"] = "polaris"  # type: ignore[assignment]
    label: str = "htex"

    num_nodes: int = 1
    """Number of nodes to request"""
    worker_init: str = ""
    """How to start a worker. Should load any modules and environments."""
    scheduler_options: str = "#PBS -l filesystems=home:eagle:grand"
    """PBS directives, pass -J for array jobs."""
    account: str
    """The account to charge compute to."""
    queue: str
    """Which queue to submit jobs to, will usually be prod."""
    walltime: str
    """Maximum job time."""
    cpus_per_node: int = 32
    """Up to 64 with multithreading."""
    cores_per_worker: float = 8
    """Number of cores per worker. Evenly distributed between GPUs."""
    available_accelerators: int = 4
    """Number of GPU to use."""
    retries: int = 1
    """Number of retries upon failure."""

    def get_parsl_config(self, run_dir: PathLike) -> Config:
        """Create a parsl configuration for running on Polaris@ALCF.

        We will launch 4 workers per node, each pinned to a different GPU.

        Parameters
        ----------
        run_dir: PathLike
            Directory in which to store Parsl run files.
        """
        return Config(
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    worker_debug=True,
                    # available_accelerators will override settings
                    # for max_workers
                    available_accelerators=self.available_accelerators,
                    cores_per_worker=self.cores_per_worker,
                    address=address_by_interface("bond0"),
                    cpu_affinity="block-reverse",
                    prefetch_capacity=0,
                    provider=PBSProProvider(
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind",
                            overrides="--depth=64 --ppn 1",
                        ),
                        account=self.account,
                        queue=self.queue,
                        select_options="ngpus=4",
                        # PBS directives: for array jobs pass '-J' option
                        scheduler_options=self.scheduler_options,
                        # Command to be run before starting a worker, such as:
                        worker_init=self.worker_init,
                        # number of compute nodes allocated for each block
                        nodes_per_block=self.num_nodes,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1,  # Increase to have more parallel jobs
                        cpus_per_node=self.cpus_per_node,
                        walltime=self.walltime,
                    ),
                ),
            ],
            run_dir=str(run_dir),
            retries=self.retries,
            app_cache=True,
        )


ComputeConfigTypes = Union[
    LocalConfig,
    WorkstationConfig,
    WorkstationV2Config,
    HybridWorkstationConfig,
    InferenceTrainWorkstationConfig,
    VistaConfig,
]
