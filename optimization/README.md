# Optimization module
## CasADi Installation
In case that command: (pip install casadi) not working in conda env, try:
```bash
conda config --add channels conda-forge
conda install casadi
```

If not working, then download the casadi package at: https://web.casadi.org/get/
and import the package as: 

> Alternatively, CasADi can be installed in Binary installation (Py36-Linux) and used with: `from sys import path` and `path.append(r"<yourpath>/casadi")`. Detailed guidance refers to: [Installing CasADi](https://github.com/casadi/casadi/wiki/InstallationInstructions#option-3-casadi-on-conda). 

```python
from sys import path
path.append(r'/home/nameit/Code_base')
import casadi  as cd
```

