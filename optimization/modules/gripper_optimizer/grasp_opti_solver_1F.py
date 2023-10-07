import casadi as cd

# from sys import path
# path.append(r'/home/smarnlab/Code_base')
# import casadi as cd

import matplotlib.pyplot as plt
import numpy as np

### One-Fingered Optimization

class GraspOptimSolver_1F:
    def __init__(self, solver_name="ipopt", gripper_type="BubbleG_1F"):
        ### Parameters 
        self.gripper_type = gripper_type
        gripperConstraints = {
            'BubbleG_1F':{'dist_min_thread':10, 'dist_max_thread':20, 'angle_sample_size':8}
        }
        self.solver_name = solver_name
        self.init_dist_min_thread = gripperConstraints[gripper_type]['dist_min_thread']
        self.init_dist_max_thread = gripperConstraints[gripper_type]['dist_max_thread']
        self.init_angle_sample_size = gripperConstraints[gripper_type]['angle_sample_size']

    def refresh_mapsfuns(self, angle_map, certainty_map):
        ### Angle and prob maps 
        xgrid_dense = np.linspace(0,300,300, endpoint=False)
        ygrid_dense = np.linspace(0,300,300, endpoint=False)
        self.M_desen_ang = np.copy(angle_map)
        self.M_desen_cer = np.copy(certainty_map)
        self.uncer_max = np.max(self.M_desen_cer)

        ang_flat = self.M_desen_ang.ravel(order='F')
        Rf_ang = cd.interpolant('Rf_ang','bspline',[xgrid_dense, ygrid_dense], ang_flat)
        cer_flat = self.M_desen_cer.ravel(order='F')
        Rf_cer = cd.interpolant('Rf_cer','bspline',[xgrid_dense, ygrid_dense], cer_flat)

        ### Initial cd Functions
        a = cd.MX.sym("a", 1, 1)
        b = cd.MX.sym("b", 1, 1)
        self.func_Rf_ang = cd.Function('func_ang', [a, b], [Rf_ang(cd.vcat([a, b]))], ['a', 'b'], ['Rf1'])
        self.func_Rf_cer = cd.Function('func_cer', [a, b], [Rf_cer(cd.vcat([a, b]))], ['a', 'b'], ['Rf2'])
        c = cd.MX.sym("c", 1, 1)
        self.angle_tocs = cd.Function('angle_tocs', [c], [cd.hcat([cd.cos(c), cd.sin(c)])])
        uu = cd.MX.sym("uu", 1, 1)
        vv = cd.MX.sym("vv", 1, 1)
        dd = cd.MX.sym("dd", 1, 1)
        aa = cd.MX.sym("aa", 1, 1)
        self.fun_mapangunc = cd.Function('fun_angunc', [uu, vv, dd, aa], \
                [0.5*cd.norm_2(self.angle_tocs(self.func_Rf_ang(uu - dd*cd.sin(aa), vv + dd*cd.cos(aa))) \
                - self.angle_tocs(aa + cd.pi))])

    def init_solver(self):
        ### init_solver
        self.opti_solver = cd.Opti()

        ### Variables
        self.u1 = self.opti_solver.variable()
        self.v1 = self.opti_solver.variable()
        self.dis = self.opti_solver.variable()

        ### Parameters
        self.dist_min_threshold = self.opti_solver.parameter()
        self.dist_max_threshold = self.opti_solver.parameter()
        self.opti_solver.set_value(self.dist_min_threshold, self.init_dist_min_thread)
        self.opti_solver.set_value(self.dist_max_threshold, self.init_dist_max_thread)

        self.obj_ori_var_threshold = self.opti_solver.parameter()
        self.obj_cer_ave_threshold = self.opti_solver.parameter()
        self.opti_solver.set_value(self.obj_ori_var_threshold, 0.3)
        self.opti_solver.set_value(self.obj_cer_ave_threshold, 0.5)

        ### Objective function
        sampled_angle = np.linspace(-cd.pi, cd.pi, self.init_angle_sample_size, endpoint=False)
        obj_cer, obj_dir = 0, 0

        for angle_i in sampled_angle:
            obj_cer += self.func_Rf_cer(self.u1 - self.dis*cd.sin(angle_i), self.v1 - self.dis*cd.cos(angle_i))
            obj_dir += self.fun_mapangunc(self.u1, self.v1, self.dis, angle_i)
        self.obj_ori_var = obj_dir / float(self.init_angle_sample_size)
        self.obj_cer_ave = obj_cer / float(self.init_angle_sample_size)

        self.objfun = 100*(self.obj_ori_var - self.obj_cer_ave) + self.dis
        self.opti_solver.minimize(self.objfun)

        ### Constraints
        range_uv = 300
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.u1, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.v1, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(self.dist_min_threshold, self.dis, self.dist_max_threshold))
        self.opti_solver.subject_to(self.obj_ori_var < self.obj_ori_var_threshold)
        self.opti_solver.subject_to(self.obj_cer_ave > self.obj_cer_ave_threshold)

    @staticmethod
    def plot_variable(x, y, d, opt_a, opti_b, opti_all):
        print("VVVVV: {:.2f}/{:.2f}/{:.2f}  OOOOO: {:.2f} // {:.2f} == ### {:.2f} ###" .format(x, y, d, opt_a, opti_b, opti_all))

    def iterate_solver(self, max_iter=200, total_intervals=5, trials_each_interval=10, rich_output=False):
        ### Solver
        s_opts = {
            # 'tol':10, 
            # 'compl_inf_tol':100,
            # 'dual_inf_tol': 10,
            'max_iter': max_iter,

            'mu_target': 1e-3,
            'acceptable_tol':1e-6, # here is what you need
            'acceptable_iter':5,
            'acceptable_dual_inf_tol':0.5,
            'acceptable_constr_viol_tol':1e-9,
            'acceptable_compl_inf_tol':1,
            'acceptable_obj_change_tol':1e-6, # here is what you need

            'print_level': 5 if rich_output else 1, 
            'print_user_options':'yes' if rich_output else 'no', 
            'print_options_documentation':'no',
            }
        
        self.opti_solver.solver(self.solver_name, {'ipopt':s_opts, 'print_time': 0})

        ### Debug settings output
        if rich_output:
            self.opti_solver.callback(lambda i: self.plot_variable(
                self.opti_solver.debug.value(self.u1), 
                self.opti_solver.debug.value(self.v1), 
                self.opti_solver.debug.value(self.dis), 
                self.opti_solver.debug.value(self.obj_ori_var), 
                self.opti_solver.debug.value(self.obj_cer_ave), 
                self.opti_solver.debug.value(self.objfun)
                )
            )
        else: self.opti_solver.callback()

        ### Start Optimize
        solutions = []
        best_solution = {'objfun':1e+10, 'state': 'F'}
        solution_succ_num = 0

        for unc_interval_i in range(total_intervals):
            uncer_favor = self.uncer_max * float(total_intervals - unc_interval_i) / (total_intervals - unc_interval_i + 2)
            index_good = np.array(np.nonzero(self.M_desen_cer > uncer_favor)).T

            ### try select [trials_each_interval] times at unc_interval_i
            for trail_i in range(trials_each_interval):
                print('\n=={}== Trial_{} selection in unc_favor: {} ========='.format(solution_succ_num, trail_i, uncer_favor))
                raint = np.random.randint(index_good.shape[0], size=1)
                point1 = index_good[raint[0]]
                init_dist = (self.init_dist_min_thread + self.init_dist_max_thread) / 2
                print("Try starting points: ", point1, init_dist)
                self.opti_solver.set_initial(self.u1, point1[0])
                self.opti_solver.set_initial(self.v1, point1[1])
                self.opti_solver.set_initial(self.dis, init_dist)

                # self.opti_solver.set_value(self.p_cer_threshold, unc_favor)
                # self.opti_solver.set_value(self.dist_min_threshold, distmin_favor)
                # self.opti_solver.set_value(self.dist_max_threshold, distmax_favor)
                # self.opti_solver.set_value(self.obj_ori_var_threshold, 0.4)
                # self.opti_solver.set_value(self.obj_cer_ave_threshold, 0.6)
                ### Start try
                try:
                    sol = self.opti_solver.solve()
                except Exception as e:
                    print("Failure occurred at Trial {} selection in unc_favor: {} ".format(trail_i, uncer_favor))
                    solution = {
                            'state':'F',
                            'dis':self.opti_solver.debug.value(self.dis), 
                            'u1':self.opti_solver.debug.value(self.u1), 
                            'v1':self.opti_solver.debug.value(self.v1), 
                            'uncer_favor':uncer_favor,
                            'obj_ori_var':self.opti_solver.debug.value(self.obj_ori_var), 
                            'obj_cer_ave':self.opti_solver.debug.value(self.obj_cer_ave), 
                            'objfun':self.opti_solver.debug.value(self.objfun)
                    }
                    if solution['dis'] < self.init_dist_max_thread and \
                        solution['obj_ori_var'] < 0.3 and solution['obj_cer_ave'] > 0.4 * self.uncer_max:
                        solution['state'] = 'S'
                else:
                    print('\n\nSuccessing: in trial: {}\n\n'.format(trail_i))
                    solution = {
                            'state':'S',
                            'dis':sol.value(self.dis),
                            'u1':sol.value(self.u1),
                            'v1':sol.value(self.v1),
                            'uncer_favor':uncer_favor,
                            'obj_ori_var':sol.value(self.obj_ori_var),
                            'obj_cer_ave':sol.value(self.obj_cer_ave), 
                            'objfun':sol.value(self.objfun)
                    }
                
                solution_stats = self.opti_solver.stats()
                if 'iterations' not in solution_stats: continue
                if rich_output: print("Solution\n", solution)
                
                obj_min = np.array(solution_stats['iterations']['obj']).min()
                solution['obj_min'] = obj_min
                solutions.append(solution)

                if (solution['state'] == 'S') and (best_solution['state'] == 'F'): best_solution = solution
                if (solution['objfun'] < best_solution['objfun']) and (best_solution['state'] == solution['state']): best_solution = solution

                if solution['state'] == 'S': solution_succ_num += 1
                if solution_succ_num > total_intervals: return solutions, best_solution
        return solutions, best_solution

    def build_grasp(self, solution):
        if solution['state'] == 'F': return [0, 0, 0, 0, 0], 'NNNNNNNNNN'

        p_u = solution['u1']
        p_v = solution['v1']
        dis = solution['dis']
        orientation = 0.0
        grsap_opening = dis
        palm_position = 0.0

        grasp_params = [p_u, p_v, orientation, grsap_opening, palm_position]
        return grasp_params, self.gripper_type

    @staticmethod
    def plot_result(certainty_map, solution = None, rgb_image=None):
        """ 
        Convenient function for visualization
        """
        # start ploting
        plt.figure("solution")
        if rgb_image is not None:
            plt.imshow(rgb_image)
            # certainty_map = np.ma.masked_where(certainty_map < 0.01, certainty_map)
            certainty_map = np.clip(certainty_map, 0, 1)
            plt.imshow(certainty_map, cmap='jet', alpha=certainty_map*0.8)
        else:
            plot = plt.matshow(certainty_map)
            plt.colorbar(plot)

        if solution['objfun'] > 1e8:
            plt.text(10, 20, 'No solution', fontdict=dict(fontsize=10, color='w', weight='bold'),
                bbox=dict(fill=True, color='k', linewidth=1, alpha=0.8))
            return plt
    
        (u1, v1, dis) = solution['u1'], solution['v1'], solution['dis']
        sta, cer, ori, obj = solution['state'], solution['obj_cer_ave'], solution['obj_ori_var'], solution['objfun']
        point_center = np.array([[u1, v1]])

        theta = np.linspace( 0 , 2 * np.pi , 20)
        uu, vv = dis * np.sin(theta) + u1, dis * np.cos(theta) + v1
        plt.plot(vv, uu, linewidth=4, color='k' if sta == 'S' else 'r', alpha=0.8)
        plt.scatter(point_center[:, 1], point_center[:, 0], s=10, color='w')
        plt.text(10, 20, '{}: {:.1f},{:.2f},{:.2f}'.format(sta, cer, ori, obj), 
            fontdict=dict(fontsize=10, color='w', weight='bold'),
            bbox=dict(fill=True, color='k', linewidth=1, alpha=0.8))
        # plt.show()
        return plt

# ###
# ### Try it now !
# ###
# gripper_optimizer = GraspOptimSolver_1F()
# gripper_optimizer.refresh_mapsfuns(angle_map, certainty_map)
# gripper_optimizer.init_solver()

# print("Start iteration ...")
# sol_pack, sol_best = gripper_optimizer.iterate_solver(max_iter=1000, total_intervals=5, trials_each_interval=3, rich_output=False)

# best_objs = []
# for sol_pack_i in sol_pack: 
#     best_objs.append(sol_pack_i['objfun'])
#     best_objs.append(sol_pack_i['obj_min'])
#     best_objs.append(sol_pack_i['state'])
#     print('\n\nsol_pack:', sol_pack_i)
# print('\n\nsol_best:', sol_best, '\n\n', best_objs)

# if(len(sol_pack) > 0): gripper_optimizer.plot_result(certainty_map, sol_best)