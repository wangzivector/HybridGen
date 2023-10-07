import casadi as cd

# from sys import path
# path.append(r'/home/smarnlab/Code_base')
# import casadi as cd

import matplotlib.pyplot as plt
import numpy as np

### Two-Fingered Optimization

class GraspOptimSolver_2F:
    def __init__(self, solver_name="ipopt", gripper_type="Robotiq_2F"):
        ### Parameters 
        gripperConstraints = {
            'Robotiq_2F':{'dist_min_thread':30, 'dist_max_thread':70},
            'FishGri_2F':{'dist_min_thread':30, 'dist_max_thread':80},
            'RochuGr_2F':{'dist_min_thread':30, 'dist_max_thread':70}
        }
        self.solver_name = solver_name
        self.gripper_type = gripper_type
        self.init_dist_min_thread = gripperConstraints[gripper_type]['dist_min_thread']
        self.init_dist_max_thread = gripperConstraints[gripper_type]['dist_max_thread']

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
        tt = cd.MX.sym("tt", 1, 1)
        self.fun_mapangdiff = cd.Function('fun_mapangdiff', [uu, vv, tt], \
                [0.5*cd.norm_2(self.angle_tocs(self.func_Rf_ang(uu, vv)) - self.angle_tocs(tt))])
        tt1 = cd.MX.sym("tt", 1, 1)
        tt2 = cd.MX.sym("tt", 1, 1)
        self.fun_angdiff = cd.Function('fun_angdiff', [tt1, tt2], \
                [0.5*cd.norm_2(self.angle_tocs(tt2) - self.angle_tocs(tt1))])
        var_u1 = cd.MX.sym('var_u1')
        var_v1 = cd.MX.sym('var_v1')
        var_u2 = cd.MX.sym('var_u2')
        var_v2 = cd.MX.sym('var_v2')
        self.distance = cd.Function('distance', [var_u1, var_v1, var_u2, var_v2], [cd.sqrt((var_u1-var_u2)**2 + (var_v1-var_v2)**2)])

    def init_solver(self):
        ### init_solver
        self.opti_solver = cd.Opti()

        ### Variables
        self.u1 = self.opti_solver.variable()
        self.v1 = self.opti_solver.variable()
        self.t1 = self.opti_solver.variable()
        self.u2 = self.opti_solver.variable()
        self.v2 = self.opti_solver.variable()
        self.t2 = self.opti_solver.variable()

        ### Parameters
        self.p_cer_threshold = self.opti_solver.parameter()
        self.opening_dist_min = self.opti_solver.parameter()
        self.opening_dist_max = self.opti_solver.parameter()
        self.opti_solver.set_value(self.p_cer_threshold, 0.1)
        self.opti_solver.set_value(self.opening_dist_min, self.init_dist_min_thread)
        self.opti_solver.set_value(self.opening_dist_max, self.init_dist_max_thread)

        # Objective function
        self.obj_dis = cd.sqrt((self.u1 - self.u2)**2 + (self.v1 - self.v2)**2)
        self.obj_cer = 0.5*(self.func_Rf_cer(self.u1, self.v1) + self.func_Rf_cer(self.u2, self.v2))
        self.obj_ang_corr = 0.5*(self.fun_mapangdiff(self.u1, self.v1, self.t1) + self.fun_mapangdiff(self.u2, self.v2, self.t2))
        self.obj_ang_anti = 1 - self.fun_angdiff(self.t2, self.t1)
        self.obj_ang = 0.5*(self.obj_ang_corr + self.obj_ang_anti)
        self.objfun = self.obj_dis + 100 * (self.obj_ang - self.obj_cer)
        # self.objfun = self.obj_dis + 100 * (- self.obj_cer)
        
        self.opti_solver.minimize(self.objfun)

        # Constraints
        range_uv = 300
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.u1, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.u2, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.v1, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.v2, range_uv))
        self.opti_solver.subject_to(self.func_Rf_cer(self.u1, self.v1) > self.p_cer_threshold) # may bad
        self.opti_solver.subject_to(self.func_Rf_cer(self.u2, self.v2) > self.p_cer_threshold) # may bad
        dist = self.distance(self.u1, self.v1, self.u2, self.v2)
        self.opti_solver.subject_to(self.opti_solver.bounded(self.opening_dist_min, dist, self.opening_dist_max))

        self.opti_solver.subject_to(self.t1==cd.atan2(self.u1-self.u2, self.v2-self.v1))
        self.opti_solver.subject_to(self.t2==cd.atan2(self.u2-self.u1, self.v1-self.v2))
        self.opti_solver.subject_to(self.func_Rf_ang(self.u1, self.v1) == self.t1)
        self.opti_solver.subject_to(self.func_Rf_ang(self.u2, self.v2) == self.t2)

    @staticmethod
    def plot_variable(x1, y1, z1, x2, y2, z2, opt_a, opti_b, opti_c, opti_all):
        print("VVVVV: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}  OOOOO: {:.2f}//{:.2f}//{:.2f} == ### {:.2f} ###"\
            .format(x1, y1, z1, x2, y2, z2, opt_a, opti_b, opti_c, opti_all))

    def iterate_solver(self, max_iter=100, total_intervals=10, trials_each_interval=20, rich_output=False):
        ### Solver
        s_opts = {
            # 'tol':10, 
            # 'compl_inf_tol':100,
            # 'dual_inf_tol': 10,
            'max_iter': max_iter,

            'mu_target': 1e-1,
            'acceptable_tol':1,
            'acceptable_iter':3,
            'acceptable_dual_inf_tol':0.5,
            'acceptable_constr_viol_tol':1e-3,
            'acceptable_compl_inf_tol':1,
            'acceptable_obj_change_tol':1e-3,

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
                self.opti_solver.debug.value(self.t1), 
                self.opti_solver.debug.value(self.u2), 
                self.opti_solver.debug.value(self.v2), 
                self.opti_solver.debug.value(self.t2), 
                self.opti_solver.debug.value(self.obj_dis), 
                self.opti_solver.debug.value(self.obj_cer),
                self.opti_solver.debug.value(self.obj_ang), 
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
                point_inits, point_angles = [], np.zeros(2)
                for try_sample_i in range(20):
                    raint = np.random.randint(index_good.shape[0], size=2)
                    point_inits = index_good[raint]
                    for index in range(point_inits.shape[0]): point_angles[index] = self.func_Rf_ang(point_inits[index][0], point_inits[index][1])
                    var_angles = np.sqrt(np.mean(np.cos(point_angles))**2 + np.mean(np.sin(point_angles))**2)
                    if var_angles < 0.2: break
                    # else: print(point_angles, 'not good, resampling....')

                # # print('point_inits', point_inits, 'index_good_sorted', index_good_sorted)

                point1 = index_good[raint[0]]
                point2 = index_good[raint[1]]
                ang1 = self.func_Rf_ang(point1[0],point1[1]) + 1e-3 # for ERR gradient
                ang2 = self.func_Rf_ang(point2[0],point2[1]) - 1e-3 # for ERR gradient
                
                print("Try starting points: ", point1, point2, ang1, ang2)
                self.opti_solver.set_initial(self.u1, point1[0])
                self.opti_solver.set_initial(self.v1, point1[1])
                self.opti_solver.set_initial(self.u2, point2[0])
                self.opti_solver.set_initial(self.v2, point2[1])
                self.opti_solver.set_initial(self.t1, ang1)
                self.opti_solver.set_initial(self.t2, ang2)

                self.opti_solver.set_value(self.p_cer_threshold, uncer_favor*0.8) # test
                # self.opti_solver.set_value(self.opening_dist_min, dist_min_favor)
                # self.opti_solver.set_value(self.opening_dist_max, dist_max_favor)

                try: 
                    sol = self.opti_solver.solve()
                except Exception as e:
                    print("Failure occurred at Trial {} selection in unc_favor: {} ".format(trail_i, uncer_favor))
                    
                    # print("POSSIBLE infeasibility")
                    # self.opti_solver.debug.show_infeasibilities()

                    solution = {
                            'state':'F',
                            'u1':self.opti_solver.debug.value(self.u1), 
                            'v1':self.opti_solver.debug.value(self.v1), 
                            't1':self.opti_solver.debug.value(self.t1), 
                            'u2':self.opti_solver.debug.value(self.u2), 
                            'v2':self.opti_solver.debug.value(self.v2), 
                            't2':self.opti_solver.debug.value(self.t2), 
                            'uncer_favor':uncer_favor,
                            'obj_dis':self.opti_solver.debug.value(self.obj_dis), 
                            'obj_cer':self.opti_solver.debug.value(self.obj_cer), 
                            'obj_ang':self.opti_solver.debug.value(self.obj_ang), 
                            'objfun':self.opti_solver.debug.value(self.objfun)}
                else:
                    print('\n\nSuccessing: in trial: {}\n\n'.format(trail_i))
                    solution = {
                            'state':'S',
                            'u1':sol.value(self.u1),
                            'v1':sol.value(self.v1),
                            't1':sol.value(self.t1),
                            'u2':sol.value(self.u2),
                            'v2':sol.value(self.v2),
                            't2':sol.value(self.t2),
                            'uncer_favor':uncer_favor,
                            'obj_dis':sol.value(self.obj_dis),
                            'obj_cer':sol.value(self.obj_cer),
                            'obj_ang':sol.value(self.obj_ang),
                            'objfun':sol.value(self.objfun)}
                # print("START DEBUG ... ")
                # print(self.opti_solver.debug.g_describe(1))
                # print(self.opti_solver.debug.x_describe(1))
                # print()
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

        u1, v1 = solution['u1'], solution['v1']
        u2, v2 = solution['u2'], solution['v2']

        p_u = (u1 + u2)/2
        p_v = (v1 + v2)/2
        
        grsap_opening = np.sqrt((u1 - u2)**2 + (v1 - v2)**2)
        orientation = np.arctan((u1 - u2)/(-(v1  - v2)))
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
        
        # unpack result here
        u1, v1 = solution['u1'], solution['v1']
        u2, v2 = solution['u2'], solution['v2']
        sta, cer, obj = solution['state'], solution['obj_cer'], solution['objfun']

        point_center = np.array([[(u1 + u2)/2, (v1 + v2)/2]])
        points = np.array([[u1, v1], [u2, v2]])
        plt.plot(points[:, 1], points[:, 0], linewidth=4, color='k' if sta == 'S' else 'r', alpha=0.8, zorder=0)
        plt.text(10, 20, '{}: {:.1f},{:.1f}'.format(sta, cer, obj), fontdict=dict(fontsize=10, color='w', weight='bold'),
            bbox=dict(fill=True, color='k', linewidth=1, alpha=0.8), zorder=5)
        plt.scatter(point_center[:, 1], point_center[:, 0], s=10, color='w', zorder=10)
        # plt.show()
        return plt


# ###
# ### Try it now !
# ###
# gripper_optimizer = GraspOptimSolver_2F()
# gripper_optimizer.refresh_mapsfuns(angle_map, certainty_map)
# gripper_optimizer.init_solver()

# print("Start iteration ...")
# sol_pack, sol_best = gripper_optimizer.iterate_solver(max_iter=300, total_intervals=5, trials_each_interval=3, rich_output=True)

# best_objs = []
# for sol_pack_i in sol_pack: 
#     best_objs.append(sol_pack_i['objfun'])
#     best_objs.append(sol_pack_i['obj_min'])
#     best_objs.append(sol_pack_i['state'])
#     print('\n\nsol_pack:', sol_pack_i)
# print('\n\nsol_best:', sol_best, '\n\n', best_objs)

# if(len(sol_pack) > 0): gripper_optimizer.plot_result(certainty_map, sol_best)