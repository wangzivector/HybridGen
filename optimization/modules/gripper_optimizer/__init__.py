"""
In case that command: (pip install casadi) not working in conda env. 
try to download package at: https://web.casadi.org/get/
and import the package as: 

from sys import path
path.append(r'/home/smarnlab/Code_base')
import casadi  as cd
"""

def GripperOptimizer(gripper_category):
    if gripper_category == "Gripper_1F":
        from .grasp_opti_solver_1F import GraspOptimSolver_1F
        return GraspOptimSolver_1F
    elif gripper_category == "Gripper_2F":
        from .grasp_opti_solver_2F import GraspOptimSolver_2F
        return GraspOptimSolver_2F
    elif gripper_category == "Gripper_3F":
        from .grasp_opti_solver_3F import GraspOptimSolver_3F
        return GraspOptimSolver_3F
    elif gripper_category == "Gripper_RQ":
        from .grasp_opti_solver_RQ import GraspOptimSolver_Robotiq3F
        return GraspOptimSolver_Robotiq3F
    elif gripper_category == "Gripper_4F":
        from .grasp_opti_solver_4F import GraspOptimSolver_4F
        return GraspOptimSolver_4F
    else:
        raise NotImplementedError("{} is not implemented yet.".format(gripper_category))
    