# Installation

### 1. Create Python environment

The first step is to create the Python environment. We tested the code on Python 3.12.

We recommend using (mini)conda for setting up an environment. To create and activate the environment, run:

```
conda create -n dynabs-jax python=3.12
conda activate dynabs
```

Install cddlib and GMP by [following the (OS-dependent) instructions here](https://pycddlib.readthedocs.io/en/latest/quickstart.html). For example, on MacOS, you can run:

```
brew install cddlib gmp
```

Then, install the dependencies within the conda environment:

```
pip install -r requirements.txt
```

Finally, install pycddlib:

```
pip install pycddlib
```

If installing pycddlib gives you an error similar to ```Cannot open include file: 'cddlib/setoper.h': No such file or directory```, then try
to [use this troubleshoot page.](https://pycddlib.readthedocs.io/en/latest/quickstart.html#installation)
On MacOS, the suggested fix is as follows:

```
env "CFLAGS=-I$(brew --prefix)/include -L$(brew --prefix)/lib" python -m pip install pycddlib
```

### 2. Install JAX

To install JAX with CUDA support via conda, run:

```
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

To instead install JAX without CUDA support, run:

```
pip install jax==0.5.3
```

### 3. Install Storm and Stormpy

We use the probabilistic model checker Storm (via its Python bindings provided by Stormpy) to compute optimal policies on interval MDP (IMDP) abstractions.
Install the following

1. Install Storm 1.9.0 (Make sure to install the required dependencies first: https://www.stormchecker.org/documentation/obtain-storm/build.html).
2. Install Pycarl 2.3.0 within the conda environment (https://moves-rwth.github.io/pycarl/)
3. Install Pycarl 2.3.0 within the conda environment (https://www.stormchecker.org/documentation/obtain-storm/build.html).

# Reproducing experiments

The experiments presented in the paper can be reproduced by running the following commands:

```
python RunFile.py --model Dubins --model_version 0
python RunFile.py --model Dubins --model_version 1
python RunFile.py --model Dubins --model_version 2
```

These commands run the Dubin's vehicle benchmark with no, 10%, and 20% parameter uncertainty, respectively.

A smaller version of the Dubin's benchmark is also available (this can be useful for faster debugging purposes):

```
python RunFile.py --model Dubins_small
```