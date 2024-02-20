# Installation

Tested on Python 3.10. The artifact can be installed and run as follows. If you only plan to use either Storm or Prism
for the model checking, you can skip the steps for the other model checker.

1. Install Storm 1.8.0 (https://www.stormchecker.org/documentation/obtain-storm/build.html).
2. Install Jax 0.4.20 (or above), with cuda support if
   desired ([instructions for installing jax with cuda enabled](https://jax.readthedocs.io/en/latest/installation.html)].
   We tested installing jax with cuda support via conda:
   ``conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia```
3. Install the dependencies via pip:
   ```pip install -r requirements.txt```
4. Install Pycarl 2.2.0 (https://moves-rwth.github.io/pycarl/) and Stormpy
   1.8.0 (https://www.stormchecker.org/documentation/obtain-storm/build.html).
5. Install Prism 4.8 (https://prismmodelchecker.org/download.php)

# Example run

The 2D drone benchmark can be run using the following commands.

1. Model checking with Prism (change prism_dir to the appropriate folder):

```python RunFile.py --model Drone2D --checker prism --prism_dir ~/Documents/SBA/prism-imc/prism/bin/prism```

2. Model checking with Storm:

```python RunFile.py --model Drone2D --checker storm```

3. Enable debugging and compare both model checking results:

```python RunFile.py --model Drone2D --debug --prism_dir ~/Documents/SBA/prism-imc/prism/bin/prism```