# Optimization module
## CasADi Installation
In case that command: (pip install casadi) not working in conda env, try:
```bash
conda config --add channels conda-forge
conda install casadi
```

If not working, then download the casadi package at: https://web.casadi.org/get/
and import the package as: 

```python
from sys import path
path.append(r'/home/nameit/Code_base')
import casadi  as cd
```
