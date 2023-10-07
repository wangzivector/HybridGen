import casadi as cd

# from sys import path
# path.append(r'/home/smarnlab/Code_base')
# import casadi as cd

import matplotlib.pyplot as plt
import numpy as np


### Three-Fingered Optimization [Robotiq-3F]

class GraspOptimSolver_Robotiq3F:
    def __init__(self, solver_name="ipopt", gripper_type="Robotiq_3F"):
        ### Parameters
        gripperConstraints = {
            'Robotiq_3F':{'palm_index_index_d':30, 'palm_index_thumb_d':30, 'disOpen':[-10, 30], 'disScis':[-13, 20], 'thumb_angs':[-0, 0], 'index_angs':[-0, 0]},
        }
        self.solver_name = solver_name
        self.gripper_type = gripper_type
        self.init_palm_index_index_d = gripperConstraints[gripper_type]['palm_index_index_d']
        self.init_palm_index_thumb_d = gripperConstraints[gripper_type]['palm_index_thumb_d']
        self.init_disOpen = gripperConstraints[gripper_type]['disOpen']
        self.init_disScis = gripperConstraints[gripper_type]['disScis']
        self.init_thumb_angs = gripperConstraints[gripper_type]['thumb_angs']
        self.init_index_angs = gripperConstraints[gripper_type]['index_angs']

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
        var_u1 = cd.MX.sym('var_u1')
        var_v1 = cd.MX.sym('var_v1')
        var_u2 = cd.MX.sym('var_u2')
        var_v2 = cd.MX.sym('var_v2')
        self.distance = cd.Function('distance', [var_u1, var_v1, var_u2, var_v2], [cd.sqrt((var_u1-var_u2)**2 + (var_v1-var_v2)**2)])
        ### Convinent functions
        var_x1 = cd.MX.sym('var_x1')
        var_y1 = cd.MX.sym('var_y1')
        var_x2 = cd.MX.sym('var_x2')
        var_y2 = cd.MX.sym('var_y2')
        self.AngleBT = cd.Function('AngleBT', [var_x1, var_y1, var_x2, var_y2], [cd.acos((var_x1*var_x2 + var_y1*var_y2)/(cd.norm_2(cd.hcat([var_x1, var_y1])) \
                * cd.norm_2(cd.hcat([var_x2, var_y2])))) * cd.sign(var_x1*var_y2 - var_x2*var_y1)], ['var_x1', 'var_y1', 'var_x2', 'var_y2'], ['AngleBT'])
        tt1 = cd.MX.sym("tt", 1, 1)
        tt2 = cd.MX.sym("tt", 1, 1)
        self.fun_angdiff = cd.Function('fun_angdiff', [tt1, tt2], [0.5*cd.norm_2(self.angle_tocs(tt2) - self.angle_tocs(tt1))])

    def init_solver(self):
        ### init_solver
        self.opti_solver = cd.Opti()

        ### Variables
        self.ut1 = self.opti_solver.variable()
        self.vt1 = self.opti_solver.variable()
        self.th1 = self.opti_solver.variable()
        self.ut2 = self.opti_solver.variable()
        self.vt2 = self.opti_solver.variable()
        self.th2 = self.opti_solver.variable()
        self.ut3 = self.opti_solver.variable() # thumb
        self.vt3 = self.opti_solver.variable() # thumb
        self.th3 = self.opti_solver.variable() # thumb
        # self.th1 = self.th3 + cd.pi
        # self.th2 = self.th3 + cd.pi

        self.disOpen = self.opti_solver.variable()
        self.disScis = self.opti_solver.variable()
        self.upo1 = self.ut1 - self.disOpen * cd.sin(self.th1)# - self.disScis * cd.sin(self.th3 - cd.pi/2) # TOBECHECK
        self.vpo1 = self.vt1 + self.disOpen * cd.cos(self.th1)# + self.disScis * cd.cos(self.th3 - cd.pi/2)
        self.upo2 = self.ut2 - self.disOpen * cd.sin(self.th2)# - self.disScis * cd.sin(self.th3 + cd.pi/2)
        self.vpo2 = self.vt2 + self.disOpen * cd.cos(self.th2)# + self.disScis * cd.cos(self.th3 + cd.pi/2)
        self.up3 = self.ut3 - self.disOpen * cd.sin(self.th3)
        self.vp3 = self.vt3 + self.disOpen * cd.cos(self.th3)

        ### Parameters
        self.unc_thread = self.opti_solver.parameter()
        self.opti_solver.set_value(self.unc_thread, 0.7) # to be changed
        self.index_index_d = self.opti_solver.parameter()
        self.index_thumb_d = self.opti_solver.parameter()
        self.opti_solver.set_value(self.index_index_d, self.init_palm_index_index_d)
        self.opti_solver.set_value(self.index_thumb_d, self.init_palm_index_thumb_d)
        self.disOpen_min = self.opti_solver.parameter()
        self.disOpen_max = self.opti_solver.parameter()
        self.opti_solver.set_value(self.disOpen_min, self.init_disOpen[0])
        self.opti_solver.set_value(self.disOpen_max, self.init_disOpen[1])
        self.disScis_min = self.opti_solver.parameter()
        self.disScis_max = self.opti_solver.parameter()
        self.opti_solver.set_value(self.disScis_min, self.init_disScis[0])
        self.opti_solver.set_value(self.disScis_max, self.init_disScis[1])

        ### Objective function
        upm, vpm = (self.upo1 + self.upo2) / 2, (self.vpo1 + self.vpo2) / 2
        u_p3m, v_p3m = (self.up3 - upm) * cd.sign(self.disOpen), (self.vp3 - vpm) * cd.sign(self.disOpen)
        u_t11, v_t11 = (self.ut1 - self.upo1), (self.vt1 - self.vpo1)
        u_t22, v_t22 = (self.ut2 - self.upo2), (self.vt2 - self.vpo2)
        u_t33, v_t33 = (self.ut3 - self.up3), (self.vt3 - self.vp3)
        ang_t1p1 = - self.AngleBT(-(-v_p3m), -(u_p3m), -v_t11, u_t11)
        ang_t2p2 = self.AngleBT(-(-v_p3m), -(u_p3m), -v_t22, u_t22)
        ang_t3p3 = self.AngleBT(-v_p3m, u_p3m, -v_t33, u_t33)

        # obj_ang_corr_1 = self.fun_angdiff(ang_t1p1, ang_t2p2)
        obj_ang_corr_0 = self.fun_angdiff(ang_t1p1, 0)
        obj_ang_corr_1 = self.fun_angdiff(ang_t2p2, 0)
        obj_ang_corr_2 = self.fun_angdiff(ang_t3p3, 0)

        self.obj_ang = (1.0/3.0)*(obj_ang_corr_0 + obj_ang_corr_1 + obj_ang_corr_2)
        self.obj_cer = (1.0/3.0)*(self.func_Rf_cer(self.ut1, self.vt1) + self.func_Rf_cer(self.ut2, self.vt2) + self.func_Rf_cer(self.ut3, self.vt3))
        # self.objfun = cd.fabs(self.disOpen) + cd.fabs(self.disScis) + 100*(self.obj_ang + (1 - self.obj_cer))
        # self.objfun = cd.fabs(self.disScis) + 100*(self.obj_ang + (1 - self.obj_cer))
        self.objfun = 100*(self.obj_ang + (1 - self.obj_cer))
        self.opti_solver.minimize(self.objfun)

        ### Constraints
        # board constraints of image 
        range_uv = 300
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.ut1, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.ut2, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.ut3, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.vt1, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.vt2, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.vt3, range_uv))
        # Distance constraints between fingertips and palm points
        self.opti_solver.subject_to(self.opti_solver.bounded(self.disOpen_min, self.disOpen, self.disOpen_max))
        self.opti_solver.subject_to(self.opti_solver.bounded(self.disScis_min, self.disScis, self.disScis_max))
        # threshold constraints from probability maps
        self.opti_solver.subject_to(self.func_Rf_cer(self.ut1, self.vt1) > self.unc_thread)
        self.opti_solver.subject_to(self.func_Rf_cer(self.ut2, self.vt2) > self.unc_thread)
        self.opti_solver.subject_to(self.func_Rf_cer(self.ut3, self.vt3) > self.unc_thread)

        # Distance constraints between palm point postitions 
        dist_indind = self.index_index_d + 2 * self.disScis
        dist_thuind = cd.sqrt((dist_indind/2)**2 + (self.index_thumb_d**2 - (self.index_index_d/2)**2))
        self.opti_solver.subject_to(self.distance(self.upo1, self.vpo1, self.upo2, self.vpo2) == dist_indind)
        self.opti_solver.subject_to(self.distance(self.upo1, self.vpo1, self.up3, self.vp3) == dist_thuind)
        self.opti_solver.subject_to(self.distance(self.up3, self.vp3, self.upo2, self.vpo2) == dist_thuind)

        # dist_epsi_thresd = 1.0
        # self.opti_solver.subject_to(cd.fabs(self.distance(self.upo1, self.vpo1, self.upo2, self.vpo2) - dist_indind) < dist_epsi_thresd)
        # self.opti_solver.subject_to(cd.fabs(self.distance(self.upo1, self.vpo1, self.up3, self.vp3) - dist_thuind) < dist_epsi_thresd)
        # self.opti_solver.subject_to(cd.fabs(self.distance(self.up3, self.vp3, self.upo2, self.vpo2) - dist_thuind) < dist_epsi_thresd)

        # Angle map - Angle match constraints
        # For consistency at pi and -pi, cos/sin operation are adopted
        self.opti_solver.subject_to(self.func_Rf_ang(self.ut1, self.vt1) == self.th1) # self.th1
        self.opti_solver.subject_to(self.func_Rf_ang(self.ut2, self.vt2) == self.th2) # self.th2
        self.opti_solver.subject_to(self.func_Rf_ang(self.ut3, self.vt3) == self.th3) # self.th3

        # Finger orientation (to palm) constraints 
        incre_bias = cd.pi/180.0 * 1
        self.opti_solver.subject_to(self.opti_solver.bounded(self.init_index_angs[0] - incre_bias, ang_t1p1, self.init_index_angs[1] + incre_bias))
        self.opti_solver.subject_to(self.opti_solver.bounded(self.init_index_angs[0] - incre_bias, ang_t2p2, self.init_index_angs[1] + incre_bias))
        self.opti_solver.subject_to(self.opti_solver.bounded(self.init_thumb_angs[0] - incre_bias, ang_t3p3, self.init_thumb_angs[1] + incre_bias))

        
    @staticmethod
    def plot_variable(ut1, vt1, th1, ut2, vt2, th2, ut3, vt3, th3, opt_a, opti_b, opti_c, opti_d, opti_all):
        print("[{:.2f}] [{:.2f}] [{:.2f}] / [{:.2f}] [{:.2f}] [{:.2f}] / [{:.2f}] [{:.2f}] [{:.2f}] opti: {:.2f}, {:.2f}, {:.2f}, {:.2f} == {:.2f}"\
                .format(ut1, vt1, th1, ut2, vt2, th2, ut3, vt3, th3, opt_a, opti_b, opti_c, opti_d, opti_all))

    def iterate_solver(self, max_iter=300, total_intervals=5, trials_each_interval=20, rich_output=False):
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
                self.opti_solver.debug.value(self.ut1),
                self.opti_solver.debug.value(self.vt1),
                self.opti_solver.debug.value(self.th1),
                self.opti_solver.debug.value(self.ut2),
                self.opti_solver.debug.value(self.vt2),
                self.opti_solver.debug.value(self.th2),
                self.opti_solver.debug.value(self.ut3),
                self.opti_solver.debug.value(self.vt3),
                self.opti_solver.debug.value(self.th3),
                self.opti_solver.debug.value(self.disOpen),
                self.opti_solver.debug.value(self.disScis),
                self.opti_solver.debug.value(self.obj_cer),
                self.opti_solver.debug.value(self.obj_ang),
                self.opti_solver.debug.value(self.objfun),
                )
            )
        else: self.opti_solver.callback()

        ### Start Optimize
        solutions = []
        best_solution = {'objfun':1e+10, 'state': 'F'}
        solution_succ_num = 0

        for unc_interval_i in range(total_intervals):
            uncer_favor = self.uncer_max * float(total_intervals - unc_interval_i) / (total_intervals - unc_interval_i + 2)
            # interval_decay_rates = np.linspace(0.9, 0.5, total_intervals, endpoint=False)
            # uncer_favor = self.uncer_max * interval_decay_rates[unc_interval_i]
            index_good = np.array(np.nonzero(self.M_desen_cer > uncer_favor)).T

            ### try select [trials_each_interval] times at unc_interval_i
            for trail_i in range(trials_each_interval):
                print('\n========= Trial_{} selection in unc_favor: {} ========='.format(trail_i, uncer_favor))
                # raint = np.random.randint(index_good.shape[0], size=3)

                point_inits, point_angles = [], np.zeros(3)
                for try_sample_i in range(20):
                    raint = np.random.randint(index_good.shape[0], size=3)
                    point_inits = index_good[raint]
                    for index in range(point_inits.shape[0]): point_angles[index] = self.func_Rf_ang(point_inits[index][0], point_inits[index][1])
                    var_angles = np.sqrt(np.mean(np.cos(point_angles))**2 + np.mean(np.sin(point_angles))**2)
                    if var_angles < 0.2: break
                    # else: print(point_angles, 'not good, resampling....')
                # # print('point_inits', point_inits, 'index_good_sorted', index_good_sorted)

                index_turn = np.array([[0, 1, 2], [1, 2, 0], [2, 1, 0]])
                for one_order in index_turn:
                    print('\n=={}== Trial_{} selection in unc_favor: {} ========= with order:'.format(solution_succ_num, trail_i, uncer_favor), one_order)
                    point1 = index_good[raint[one_order[0]]]
                    point2 = index_good[raint[one_order[1]]]
                    point3 = index_good[raint[one_order[2]]]
                    ang1 = self.func_Rf_ang(point1[0],point1[1]) + 1e-2 # for ERR gradient
                    ang2 = self.func_Rf_ang(point2[0],point2[1]) - 1e-2 # for ERR gradient
                    ang3 = self.func_Rf_ang(point3[0],point3[1]) + 1e-3 # for ERR gradient
                    
                    print("Try starting points: ", point1, point2, point3, ang1, ang2, ang3)
                    self.opti_solver.set_initial(self.ut1, point1[0])
                    self.opti_solver.set_initial(self.vt1, point1[1])
                    self.opti_solver.set_initial(self.ut2, point2[0])
                    self.opti_solver.set_initial(self.vt2, point2[1])
                    self.opti_solver.set_initial(self.ut3, point3[0])
                    self.opti_solver.set_initial(self.vt3, point3[1])
                    self.opti_solver.set_initial(self.th1, ang1)
                    self.opti_solver.set_initial(self.th2, ang2)
                    self.opti_solver.set_initial(self.th3, ang3)
                    init_disOpen_midr = 0.8*(self.init_disOpen[0] + np.random.rand(1)[0] * (self.init_disOpen[1] - self.init_disOpen[0]))
                    init_disScis_midr = 0.8*(self.init_disScis[0] + np.random.rand(1)[0] * (self.init_disScis[1] - self.init_disScis[0]))
                    self.opti_solver.set_initial(self.disOpen, init_disOpen_midr)
                    self.opti_solver.set_initial(self.disScis, init_disScis_midr)
                    self.opti_solver.set_value(self.unc_thread, 0.8 * uncer_favor)
                    # self.opti_solver.set_value(self.dis_min, dis_min_favor)
                    # self.opti_solver.set_value(self.dis_max, dis_max_favor)

                    try: 
                        sol = self.opti_solver.solve()
                    except Exception as e:
                        print("Failure occurred at Trial {} selection in unc_favor: {} ".format(trail_i, uncer_favor))
                        
                        # print("POSSIBLE infeasibility")
                        # self.opti_solver.debug.show_infeasibilities()

                        solution = {
                            'state':'F',
                            'uncer_favor':uncer_favor,
                            'ut1':self.opti_solver.debug.value(self.ut1), 
                            'ut2':self.opti_solver.debug.value(self.ut2), 
                            'ut3':self.opti_solver.debug.value(self.ut3), 
                            'vt1':self.opti_solver.debug.value(self.vt1), 
                            'vt2':self.opti_solver.debug.value(self.vt2), 
                            'vt3':self.opti_solver.debug.value(self.vt3), 
                            'th1':self.opti_solver.debug.value(self.th1), 
                            'th2':self.opti_solver.debug.value(self.th2), 
                            'th3':self.opti_solver.debug.value(self.th3), 
                            'disOpen':self.opti_solver.debug.value(self.disOpen), 
                            'disScis':self.opti_solver.debug.value(self.disScis), 
                            'obj_cer':self.opti_solver.debug.value(self.obj_cer), 
                            'obj_ang':self.opti_solver.debug.value(self.obj_ang), 
                            'objfun':self.opti_solver.debug.value(self.objfun),
                    }
                    else:
                        print('\n\n Successing: in trial: {}\n\n'.format(trail_i))
                        solution = {
                            'state':'S',
                            'uncer_favor':uncer_favor,
                            'ut1':sol.value(self.ut1),
                            'ut2':sol.value(self.ut2),
                            'ut3':sol.value(self.ut3),
                            'vt1':sol.value(self.vt1),
                            'vt2':sol.value(self.vt2),
                            'vt3':sol.value(self.vt3),
                            'th1':sol.value(self.th1),
                            'th2':sol.value(self.th2),
                            'th3':sol.value(self.th3),
                            'disOpen':sol.value(self.disOpen),
                            'disScis':sol.value(self.disScis),
                            'obj_cer':sol.value(self.obj_cer), 
                            'obj_ang':sol.value(self.obj_ang), 
                            'objfun':sol.value(self.objfun),
                        }
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
                    if solution_succ_num > total_intervals/2: return solutions, best_solution
        return solutions, best_solution

    def build_grasp(self, solution):
        if solution['state'] == 'F': return [0, 0, 0, 0, 0], 'NNNNNNNNNN'

        # unpack result here
        ut1, vt1, th1 = solution['ut1'], solution['vt1'], solution['th1']
        ut2, vt2, th2 = solution['ut2'], solution['vt2'], solution['th2']
        ut3, vt3, th3 = solution['ut3'], solution['vt3'], solution['th3']
        disOpen = solution['disOpen']
        disScis = solution['disScis']
        up3 = ut3 - disOpen * np.sin(th3)
        vp3 = vt3 + disOpen * np.cos(th3)
        del_open_u1 = - disOpen * np.sin(th1)
        del_open_v1 = + disOpen * np.cos(th1)
        del_open_u2 = - disOpen * np.sin(th2)
        del_open_v2 = + disOpen * np.cos(th2)
        upo1 = ut1 + del_open_u1
        vpo1 = vt1 + del_open_v1
        upo2 = ut2 + del_open_u2
        vpo2 = vt2 + del_open_v2

        del_indind_u, del_indind_v = upo1 - upo2, vpo2 - vpo1
        unit_del_indind_u = del_indind_u / np.sqrt(del_indind_u**2 + del_indind_v**2)
        unit_del_indind_v = del_indind_v / np.sqrt(del_indind_u**2 + del_indind_v**2)
        del_scis_u1 = - disScis * unit_del_indind_u
        del_scis_v1 = + disScis * unit_del_indind_v
        del_scis_u2 = + disScis * unit_del_indind_u
        del_scis_v2 = - disScis * unit_del_indind_v

        up1 = upo1 + del_scis_u1
        vp1 = vpo1 + del_scis_v1
        up2 = upo2 + del_scis_u2
        vp2 = vpo2 + del_scis_v2

        p_u, p_v = (up1 + up2 + up3)/3, (vp1 + vp2 + vp3)/3
        orientation = th3
        grsap_opening = (disOpen - self.init_disOpen[0])
        palm_position = (disScis - self.init_disScis[0])/(disOpen - self.init_disOpen[0])
        grasp_params = [p_u, p_v, orientation, grsap_opening, palm_position]
        return grasp_params, self.gripper_type

    @staticmethod
    def plot_result(certainty_map, solution = None, rgb_image=None):
        """
        Convenient function for visualization
        """
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
        ut1, vt1, th1 = solution['ut1'], solution['vt1'], solution['th1']
        ut2, vt2, th2 = solution['ut2'], solution['vt2'], solution['th2']
        ut3, vt3, th3 = solution['ut3'], solution['vt3'], solution['th3']
        disOpen = solution['disOpen']
        disScis = solution['disScis']
        up3 = ut3 - disOpen * np.sin(th3)
        vp3 = vt3 + disOpen * np.cos(th3)
        del_open_u1 = - disOpen * np.sin(th1)
        del_open_v1 = + disOpen * np.cos(th1)
        del_open_u2 = - disOpen * np.sin(th2)
        del_open_v2 = + disOpen * np.cos(th2)
        upo1 = ut1 + del_open_u1
        vpo1 = vt1 + del_open_v1
        upo2 = ut2 + del_open_u2
        vpo2 = vt2 + del_open_v2

        del_indind_u, del_indind_v = upo1 - upo2, vpo2 - vpo1
        unit_del_indind_u = del_indind_u / np.sqrt(del_indind_u**2 + del_indind_v**2)
        unit_del_indind_v = del_indind_v / np.sqrt(del_indind_u**2 + del_indind_v**2)
        del_scis_u1 = - disScis * unit_del_indind_u
        del_scis_v1 = + disScis * unit_del_indind_v
        del_scis_u2 = + disScis * unit_del_indind_u
        del_scis_v2 = - disScis * unit_del_indind_v

        up1 = upo1 + del_scis_u1
        vp1 = vpo1 + del_scis_v1
        up2 = upo2 + del_scis_u2
        vp2 = vpo2 + del_scis_v2

        points_palms = np.array([[up1, vp1], [up2, vp2], [up3, vp3], [up1, vp1]])
        points_tp1 = np.array([[ut1, vt1], [upo1, vpo1]])
        points_tp2 = np.array([[ut2, vt2], [upo2, vpo2]])
        points_tp3 = np.array([[ut3, vt3], [up3, vp3]])
        points_sp1 = np.array([[up1, vp1], [upo1, vpo1]])
        points_sp2 = np.array([[up2, vp2], [upo2, vpo2]])

        point_center = np.array([[(up1 + up2 + up3)/3, (vp1 + vp2 + vp3)/3]])
        sta, cer, obj = solution['state'], solution['obj_cer'], solution['objfun']
        # start ploting
        plt.plot(points_palms[:, 1], points_palms[:, 0], '--', linewidth=2,  color='b', zorder=0)
        plt.plot(points_tp1[:, 1], points_tp1[:, 0], linewidth=4, color='k' if sta == 'S' else 'r', zorder=5)
        plt.plot(points_tp2[:, 1], points_tp2[:, 0], linewidth=4, color='k' if sta == 'S' else 'r', zorder=5)
        plt.plot(points_tp3[:, 1], points_tp3[:, 0], linewidth=4, color='k' if sta == 'S' else 'r', zorder=5)
        plt.plot(points_sp1[:, 1], points_sp1[:, 0], linewidth=4, color='g' if sta == 'S' else 'r', zorder=8)
        plt.plot(points_sp2[:, 1], points_sp2[:, 0], linewidth=4, color='g' if sta == 'S' else 'r', zorder=8)
        plt.text(10, 20, '{}: {:.1f},{:.1f}'.format(sta, cer, obj), fontdict=dict(fontsize=10, color='w', weight='bold'),
            bbox=dict(fill=True, color='k', linewidth=1, alpha=0.8))
        plt.scatter(point_center[:, 1], point_center[:, 0], s = 10, color='w', zorder=10)
        # plt.show()
        return plt
    

# ###
# ### Try it now !
# ###
# gripper_optimizer = GraspOptimSolver_Robotiq3F(gripper_type='Robotiq3F')
# gripper_optimizer.refresh_mapsfuns(angle_map, certainty_map)
# gripper_optimizer.init_solver()

# sol_pack, sol_final, sol_best = gripper_optimizer.iterate_solver(max_iter=200, total_intervals=2, trials_each_interval=20, rich_output=False)

# # for sol_pack_i in sol_pack: gripper_optimizer.plot_result(angle_map, sol_pack_i)

# best_objs = []
# for sol_pack_i in sol_pack: 
#     best_objs.append(sol_pack_i['objfun'])
#     best_objs.append(sol_pack_i['obj_min'])
#     best_objs.append(sol_pack_i['state'])
#     print('\n\nsol_pack:', sol_pack_i)

# print('\n\nsol_best:', sol_best, '\n\n', best_objs)
# if(len(sol_pack) > 0): gripper_optimizer.plot_result(angle_map, sol_final)
# print('\n\nsol_final:', sol_final)