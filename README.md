# SimAgent

This project is an implementation of Parsl in LangChain/LangGraph tool calling to the computing resource, such as GPU clusters and high performance computing platform. The examples were tested with the molecular dynamics simulation using `OpenMM` and the large language model (LLM) using `gpt-4o-mini` from OpenAI.

## Environment Setup

The base conda environment can be set up with the `yaml` file.
```bash
conda env create -f env.yml
```
The main dependencies are `LangGraph` for the LLM agent framework and `OpenMM` for the molecular simulation.


## Getting Started

The test run with a local PDB file can be run with
```python
python examples/sim_team.py
```

The Polaris run is set up with
```python
```

## Contributing

This project welcomes contributions and suggestions. For details, visit the repository's [Contributor License Agreement (CLA)](https://cla.opensource.microsoft.com) and [Code of Conduct](https://opensource.microsoft.com/codeofconduct/) pages.
