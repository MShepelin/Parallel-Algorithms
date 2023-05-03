# Computing Rank of Sparse Binary Matrix on GPU

We propose a project to solve sparse binary matrix rank computation problem on Graphics Processing Units (GPUs). This problem is not just interesting in itself, it also has a practical application in Topological Data Analysis.

This repository rovides a Python package that implements our solution. It serves as a wrapper for CUDA C DLL. To create it we aggregated modern approaches to processing sparse matrices and implementing parallel algorithms.

## Installation

Execute the command bellow inside "src" folder to add parallelrank package to your Python environment.

```bash
python setup.py install
```

## Limitations

To use this program you need a GPU on CUDA with Compuation Capability 3.0 or higher. To check whether you meet this requirement
please refer to [this link](https://developer.nvidia.com/cuda-gpus#compute) where you can click on sections to extend them and
find your model of GPU.

**WARNING**: Currently, this program doesn't check the load of your GPU or even whether you have a GPU. If you run 
computations on a large matrix (with minimum side more than 10.000), we recommend you to run them in a separate process
(not inside Jupyter cell). If you do, you can encounter an OS error in case memory gets overpopulated. If you don't
the process may seem to run infinitely, although it is not. 

It's also important to notice that Keyboard Interrupt doesn't work during computation. This is another reason to
run this library in a separate process that can be killed without killing the main kernel you are working with.

## Tests

After you install our package you can test it on random matrices with different sizes using a command below inside the repository folder.

```bash
python -m pytest -v tests/test_rank_computation.py
```

## Performance measurement

This repository includes a simple script you can use to measure performance of this library. 

1) Clone the project 
2) Install numpy, scipy, tqdm, matplotlib, tensorflow with pip
3) Install the package
4) Run "experiment.sh" in your shell. It will produce parallelrank_perfs.zip
5) You can extract the archive in the repository folder and use "scripts/build_graphs.ipynb" to see results.

Remember that results may differ between systems and GPU models.

## Literature
1. NVIDIA (2023) CUDA Toolkit Documentation. Available at: https://docs.nvidia.com/cuda/ (Accessed: 20 February 2023).
2. Sanders, J. and Kandrot, E., 2010. CUDA by example: an introduction to general-purpose GPU programming. Addison-Wesley Professional.
3. Hussain, M.T., Abhishek, G.S., Buluc¸, A. and Azad, A., 2022, May. Parallel Algorithms for Adding a Collection of Sparse Matrices. In 2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW) (pp. 285-294). IEEE.
4. Galiano, V., Migall´on, H., Migall´on, V. and Penad´es, J., 2012. GPUbased parallel algorithms for sparse nonlinear systems. Journal of Parallel and Distributed Computing, 72(9), pp.1098-1105.
5. Bell, N. and Garland, M., 2008. Efficient sparse matrix-vector multiplication on CUDA (Vol. 2, No. 5). Nvidia Technical Report NVR-2008-004, Nvidia Corporation.
6. Sherman, A.H., 1978. Algorithms for sparse Gaussian elimination with partial pivoting. ACM Transactions on Mathematical Software (TOMS), 4(4), pp.330-338.
7. TensorFlow (2023) API Documentation. Available at: https://www.tensorflow.org/api docs/ (Accessed: 19 February 2023).
8. NumPy (2023) NumPy documentation. Available at: https://numpy.org/doc/stable/ (Accessed: 19 February 2023).
9. Von Br¨omssen, E., 2021. Computing persistent homology in parallel with a functional language.
10. Yang, C., Buluc¸, A. and Owens, J.D., 2018, August. Design principles for sparse matrix multiplication on the gpu. In Euro-Par 2018: Parallel Processing: 24th International Conference on Parallel and Distributed Computing, Turin, Italy, August 27-31, 2018, Proceedings (pp. 672-687). Cham: Springer International Publishing.
11. Morozov, D. and Nigmetov, A., 2020, July. Towards lockfree persistent homology. In Proceedings of the 32nd ACM Symposium
12. Bauer, U., 2021. Ripser: efficient computation of Vietoris–Rips persistence barcodes. Journal of Applied and Computational Topology, 5(3), pp.391-423.
13. Bauer, U., Talha Bin Masood, Giunti B., Houry G., Kerber M. and Rathod A., 2022. Keeping it sparse: Computing Persistent Homology revisited. arXiv:2211.09075 [cs.CG].
14. Morozov, D. and Nigmetov, A., 2020. Towards Lockfree Persistent Homology. Association for Computing Machinery. New York, NY, United States.
15. Zhang, S., Xiao, M. and Wang, H., 2020. GPU-accelerated computation of Vietoris-Rips persistence barcodes. arXiv preprint arXiv:2003.07989.
