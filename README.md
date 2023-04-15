# Computing Rank of Sparse Binary Matrix on GPU

We propose a project to solve sparse binary matrix rank computation problem on Graphics processing units (GPUs). Although this problem is interesting in itself, it has a practical application in Topological Data Analysis. We aggregate modern approaches to processing sparse matrices and implementing parallel algorithms. We describe challenges of developing for CUDA (Compute Unified Device Architecture) and provide a design for our Python package that serves as a wrapper for CUDA C DLL. We also provide a plan to improve this technology using features of CUDA, better memory access patterns, caches, lower amount of global memory operations.

## Installation

Execute the command bellow inside "src" folder to add parallelrank package to your Python environment.

```bash
python setup.py install
```

## Tests

After you install our package you can test it on random matrices with different sizes using a command below inside the repository folder.

```
pytest -v tests/test_rank_computation.py
```

