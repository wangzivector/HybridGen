import casadi as cd

# from sys import path
# path.append(r'/home/smarnlab/Code_base')
# import casadi as cd

import matplotlib.pyplot as plt
import numpy as np

### Four-Fingered Optimization

class GraspOptimSolver_4F:
    def __init__(self, solver_name="ipopt", gripper_type="FishGrip4F"):
        ### Parameters
        gripperConstraints = {
            'FishGri_4F':{'palm_dists':[30, 30, 30*np.sqrt(2), 30*np.sqrt(2)], 'dis_open':[-15, 30], 'cor_ang':0, 'cor_line_id':['14', '23']},
            'RochuGr_4F':{'palm_dists':[30, 30, 30*np.sqrt(2), 30*np.sqrt(2)], 'dis_open':[-15, 30], 'cor_ang':np.pi/2, 'cor_line_id':['13', '24']},
        }
        self.solver_name = solver_name
        self.gripper_type = gripper_type
        self.init_palm_dists = gripperConstraints[gripper_type]['palm_dists']
        self.init_dis_open = gripperConstraints[gripper_type]['dis_open']
        self.init_cor_ang = gripperConstraints[gripper_type]['cor_ang']
        self.init_cor_line_id = gripperConstraints[gripper_type]['cor_line_id']

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
        uu = cd.MX.sym("uu", 1, 1)
        vv = cd.MX.sym("vv", 1, 1)
        tt = cd.MX.sym("tt", 1, 1)
        self.fun_mapangdiff = cd.Function('fun_mapangdiff', [uu, vv, tt], \
                [0.5*cd.norm_2(self.angle_tocs(self.func_Rf_ang(uu, vv)) - self.angle_tocs(tt) + 1e-6)])
        tt1 = cd.MX.sym("tt1", 1, 1)
        tt2 = cd.MX.sym("tt2", 1, 1)
        self.fun_angdiff = cd.Function('fun_angdiff', [tt1, tt2], [0.5*cd.norm_2(self.angle_tocs(tt2) - self.angle_tocs(tt1) + 1e-6)])
        # self.fun_angdiff_crosssin = cd.Function('fun_angdiff', [tt1, tt2], [cd.fabs(self.angle_tocs(tt1)[0] *self.angle_tocs(tt2)[1] - self.angle_tocs(tt2)[0] *self.angle_tocs(tt1)[1])])

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
        self.ut3 = self.opti_solver.variable()
        self.vt3 = self.opti_solver.variable()
        self.th3 = self.opti_solver.variable()
        self.ut4 = self.opti_solver.variable()
        self.vt4 = self.opti_solver.variable()
        self.th4 = self.opti_solver.variable()
        self.dis_open = self.opti_solver.variable()

        self.up1 = self.ut1 - self.dis_open * cd.sin(self.th1)
        self.vp1 = self.vt1 + self.dis_open * cd.cos(self.th1)
        self.up2 = self.ut2 - self.dis_open * cd.sin(self.th2)
        self.vp2 = self.vt2 + self.dis_open * cd.cos(self.th2)
        self.up3 = self.ut3 - self.dis_open * cd.sin(self.th3)
        self.vp3 = self.vt3 + self.dis_open * cd.cos(self.th3)
        self.up4 = self.ut4 - self.dis_open * cd.sin(self.th4)
        self.vp4 = self.vt4 + self.dis_open * cd.cos(self.th4)

        ### Parameters
        self.unc_thread = self.opti_solver.parameter()
        self.opti_solver.set_value(self.unc_thread, 0.7) # to be changed
        self.palm_d12 = self.opti_solver.parameter()
        self.palm_d23 = self.opti_solver.parameter()
        self.palm_d13 = self.opti_solver.parameter()
        self.palm_d24 = self.opti_solver.parameter()
        self.opti_solver.set_value(self.palm_d12, self.init_palm_dists[0])
        self.opti_solver.set_value(self.palm_d23, self.init_palm_dists[1])
        self.opti_solver.set_value(self.palm_d13, self.init_palm_dists[2])
        self.opti_solver.set_value(self.palm_d24, self.init_palm_dists[3])
        self.dis_open_min = self.opti_solver.parameter()
        self.dis_open_max = self.opti_solver.parameter()
        self.opti_solver.set_value(self.dis_open_min, self.init_dis_open[0])
        self.opti_solver.set_value(self.dis_open_max, self.init_dis_open[1])

        ### Objective function
        u_tp1, v_tp1 = (self.ut1 - self.up1), (self.vt1 - self.vp1)
        u_tp2, v_tp2 = (self.ut2 - self.up2), (self.vt2 - self.vp2)
        u_p13, v_p13 = (self.up1 - self.up3), (self.vp1 - self.vp3)
        u_p14, v_p14 = (self.up1 - self.up4), (self.vp1 - self.vp4)
        u_p23, v_p23 = (self.up2 - self.up3), (self.vp2 - self.vp3)
        u_p24, v_p24 = (self.up2 - self.up4), (self.vp2 - self.vp4)
        vector_bank = {
            '13': [u_p13*cd.sign(self.dis_open), v_p13*cd.sign(self.dis_open)],
            '14': [u_p14*cd.sign(self.dis_open), v_p14*cd.sign(self.dis_open)],
            '23': [u_p23*cd.sign(self.dis_open), v_p23*cd.sign(self.dis_open)],
            '24': [u_p24*cd.sign(self.dis_open), v_p24*cd.sign(self.dis_open)],
        }
        self.ang_tp1p_a = self.AngleBT(-vector_bank[self.init_cor_line_id[0]][1], vector_bank[self.init_cor_line_id[0]][0], -v_tp1, u_tp1)
        self.ang_tp2p_b = self.AngleBT(-vector_bank[self.init_cor_line_id[1]][1], vector_bank[self.init_cor_line_id[1]][0], -v_tp2, u_tp2)
        
        obj_ang_corr_1 = self.fun_angdiff(self.ang_tp1p_a, 0)
        obj_ang_corr_2 = self.fun_angdiff(self.ang_tp2p_b, 0)
        
        obj_ang_anti_1 = cd.norm_2(self.angle_tocs(self.th1) + self.angle_tocs(self.th3))
        obj_ang_anti_2 = cd.norm_2(self.angle_tocs(self.th2) + self.angle_tocs(self.th4))
        # obj_ang_anti_1 = self.fun_angdiff(self.th1, self.th3 + cd.pi)
        # obj_ang_anti_2 = self.fun_angdiff(self.th2, self.th4 + cd.pi)
        # obj_ang_corr_3 = self.fun_angdiff(self.th1, self.th2 + self.init_cor_ang)
        # obj_ang_corr_4 = self.fun_angdiff(self.th3, self.th4 + self.init_cor_ang)

        self.obj_anb = (1.0/2.0)*(obj_ang_corr_1 + obj_ang_corr_2)
        # self.obj_ang = (1.0/4.0)*(obj_ang_anti_1 + obj_ang_anti_2 + obj_ang_corr_3 + obj_ang_corr_4)
        self.obj_ang = (1.0/2.0)*(obj_ang_anti_1 + obj_ang_anti_2)
        self.obj_cer = (1.0/4.0)*(self.func_Rf_cer(self.ut1, self.vt1) + self.func_Rf_cer(self.ut2, self.vt2) 
            + self.func_Rf_cer(self.ut3, self.vt3) + self.func_Rf_cer(self.ut4, self.vt4))
        # self.objfun = cd.fabs(self.dis_open) + 100*(self.obj_ang + self.obj_anb + (1 - self.obj_cer))
        self.objfun = 100*(self.obj_anb + (1 - self.obj_cer))
        # self.objfun = 100*(self.obj_ang + self.obj_anb + (1 - self.obj_cer))
        self.opti_solver.minimize(self.objfun)

        ### Constraints
        # board constraints of image 
        range_uv = 300
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.ut1, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.ut2, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.ut3, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.ut4, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.vt1, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.vt2, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.vt3, range_uv))
        self.opti_solver.subject_to(self.opti_solver.bounded(0, self.vt4, range_uv))
        # Distance constraints between palm point postitions 

        # self.opti_solver.subject_to(self.distance(self.up1, self.vp1, self.up2, self.vp2) == self.palm_d12)
        # self.opti_solver.subject_to(self.distance(self.up2, self.vp2, self.up3, self.vp3) == self.palm_d23)
        # self.opti_solver.subject_to(self.distance(self.up3, self.vp3, self.up4, self.vp4) == self.palm_d12)
        # self.opti_solver.subject_to(self.distance(self.up4, self.vp4, self.up1, self.vp1) == self.palm_d23)
        # self.opti_solver.subject_to(self.distance(self.up1, self.vp1, self.up3, self.vp3) == self.palm_d13)
        # self.opti_solver.subject_to(self.distance(self.up2, self.vp2, self.up4, self.vp4) == self.palm_d24)

        dist_epsi_thresd = 1
        self.opti_solver.subject_to(cd.fabs(self.distance(self.up1, self.vp1, self.up2, self.vp2) - self.palm_d12) < dist_epsi_thresd)
        self.opti_solver.subject_to(cd.fabs(self.distance(self.up2, self.vp2, self.up3, self.vp3) - self.palm_d23) < dist_epsi_thresd)
        self.opti_solver.subject_to(cd.fabs(self.distance(self.up3, self.vp3, self.up4, self.vp4) - self.palm_d12) < dist_epsi_thresd)
        self.opti_solver.subject_to(cd.fabs(self.distance(self.up4, self.vp4, self.up1, self.vp1) - self.palm_d23) < dist_epsi_thresd)
        self.opti_solver.subject_to(cd.fabs(self.distance(self.up1, self.vp1, self.up3, self.vp3) - self.palm_d13) < dist_epsi_thresd)
        self.opti_solver.subject_to(cd.fabs(self.distance(self.up2, self.vp2, self.up4, self.vp4) - self.palm_d24) < dist_epsi_thresd)

        # Distance constraints between fingertips and palm points
        self.opti_solver.subject_to(self.opti_solver.bounded(self.dis_open_min, self.dis_open, self.dis_open_max))
        # threshold constraints from probability maps
        self.opti_solver.subject_to(self.func_Rf_cer(self.ut1, self.vt1) > self.unc_thread)
        self.opti_solver.subject_to(self.func_Rf_cer(self.ut2, self.vt2) > self.unc_thread)
        self.opti_solver.subject_to(self.func_Rf_cer(self.ut3, self.vt3) > self.unc_thread)
        self.opti_solver.subject_to(self.func_Rf_cer(self.ut4, self.vt4) > self.unc_thread)
        # Angle map - Angle match constraints
        # For consistency at pi and -pi, cos/sin operation are adopted
        
        self.opti_solver.subject_to(self.func_Rf_ang(self.ut1, self.vt1) == self.th1) # self.th1
        self.opti_solver.subject_to(self.func_Rf_ang(self.ut2, self.vt2) == self.th2) # self.th2
        self.opti_solver.subject_to(self.func_Rf_ang(self.ut3, self.vt3) == self.th3) # self.th3
        self.opti_solver.subject_to(self.func_Rf_ang(self.ut4, self.vt4) == self.th4) # self.th4

        cs_epsi_thresd = 5e-2 # 5e-2: 3 degs
        # vertical to each other
        self.opti_solver.subject_to(cd.fabs(self.angle_tocs(self.th1) + self.angle_tocs(self.th3)) < cs_epsi_thresd)
        self.opti_solver.subject_to(cd.fabs(self.angle_tocs(self.th2) + self.angle_tocs(self.th4)) < cs_epsi_thresd)

        # Finger orientation (to palm) constraints
        # cs_epsi_thresd = 5e-2
        # self.opti_solver.subject_to(self.fun_angdiff(self.th1, self.th3 + cd.pi) < cs_epsi_thresd)
        # self.opti_solver.subject_to(self.fun_angdiff(self.th2, self.th4 + cd.pi) < cs_epsi_thresd)
        # self.opti_solver.subject_to(self.fun_angdiff(self.th1, self.th2 + self.init_cor_ang) < cs_epsi_thresd)
        # self.opti_solver.subject_to(self.fun_angdiff(self.th3, self.th4 + self.init_cor_ang) < cs_epsi_thresd)

        # self.opti_solver.subject_to(self.fun_angdiff_crosssin(self.ang_tp1p14, 0) < cs_epsi_thresd)
        # self.opti_solver.subject_to(self.fun_angdiff_crosssin(self.ang_tp2p23, 0) < cs_epsi_thresd)
        incre_bias = cd.pi/180.0 * 3
        self.opti_solver.subject_to(self.opti_solver.bounded(- incre_bias, self.ang_tp1p_a, + incre_bias))
        self.opti_solver.subject_to(self.opti_solver.bounded(- incre_bias, self.ang_tp2p_b, + incre_bias))
        # self.opti_solver.subject_to(self.fun_angdiff(self.ang_tp1p14, 0) < cs_epsi_thresd)
        # self.opti_solver.subject_to(self.fun_angdiff(self.ang_tp2p23, 0) < cs_epsi_thresd)

    @staticmethod
    def plot_variable(ut1, vt1, th1, ut2, vt2, th2, ut3, vt3, th3, ut4, vt4, th4, opt_a, opti_b, opti_c, opti_d, opti_all):
        print("[{:.2f}] [{:.2f}] [{:.2f}] / [{:.2f}] [{:.2f}] [{:.2f}] / [{:.2f}] [{:.2f}] [{:.2f}] / [{:.2f}] [{:.2f}] [{:.2f}] opti: {:.2f}, {:.2f}, {:.2f}, {:.2f} == {:.2f}"\
                .format(ut1, vt1, th1, ut2, vt2, th2, ut3, vt3, th3, ut4, vt4, th4, opt_a, opti_b, opti_c, opti_d, opti_all))

    def iterate_solver(self, max_iter=300, total_intervals=5, trials_each_interval=20, rich_output=False):
        ### Solver
        s_opts = {
            # 'tol':10, 
            # 'compl_inf_tol':100,
            # 'dual_inf_tol': 10,
            'max_iter': max_iter,

            'mu_target': 1e-1,
            'acceptable_tol':1, #50,
            'acceptable_iter':3,
            'acceptable_dual_inf_tol':1, #50,
            'acceptable_constr_viol_tol':1e-2,
            'acceptable_compl_inf_tol':1,
            'acceptable_obj_change_tol':1e-2,

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
                self.opti_solver.debug.value(self.ut4),
                self.opti_solver.debug.value(self.vt4),
                self.opti_solver.debug.value(self.th4),
                self.opti_solver.debug.value(self.dis_open),
                self.opti_solver.debug.value(self.obj_cer),
                self.opti_solver.debug.value(self.obj_ang),
                self.opti_solver.debug.value(self.obj_anb),
                self.opti_solver.debug.value(self.objfun),
                )
            )
        else: self.opti_solver.callback()

        ### Start Optimize
        solutions = []
        best_solution = {'objfun':1e+10, 'state': 'F'}
        solution_succ_num = 0

        for unc_interval_i in range(total_intervals):
            uncer_favor = self.uncer_max * float(total_intervals - unc_interval_i) / (total_intervals - unc_interval_i + 3)
            # interval_decay_rates = np.linspace(decrease_rate, 0.5, total_intervals, endpoint=False)
            # uncer_favor = self.uncer_max * interval_decay_rates[unc_interval_i]
            index_good = np.array(np.nonzero(self.M_desen_cer > uncer_favor)).T

            ### try select [trials_each_interval] times at unc_interval_i
            for trail_i in range(trials_each_interval):
                print('\n========= Trial_{} selection in unc_favor: {} ========='.format(trail_i, uncer_favor))
                point_inits, point_angles = [], np.zeros(4)
                for try_sample_i in range(20):
                    raint = np.random.randint(index_good.shape[0], size=4)
                    point_inits = index_good[raint]
                    for index in range(point_inits.shape[0]): point_angles[index] = self.func_Rf_ang(point_inits[index][0], point_inits[index][1])
                    var_angles = np.sqrt(np.mean(np.cos(point_angles))**2 + np.mean(np.sin(point_angles))**2)
                    if var_angles < 0.2: break
                    # else: print(point_angles, 'not good, resampling....')
                point_argsort = np.argsort(point_angles)
                index_good_sorted = point_inits[point_argsort[::-1]]

                # print('point_inits', point_inits, 'index_good_sorted', index_good_sorted)

                # index_turn = np.array([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]])
                index_turn = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
                for one_order in index_turn:
                    print('\n =={}== Trial_{} selection in unc_favor: {} ========= with order:'.format(solution_succ_num, trail_i, uncer_favor), one_order)
                    point1 = index_good_sorted[one_order[0]]
                    point2 = index_good_sorted[one_order[1]]
                    point3 = index_good_sorted[one_order[2]]
                    point4 = index_good_sorted[one_order[3]]
                    ang1 = self.func_Rf_ang(point1[0],point1[1]) + 1e-3 # for ERR gradient
                    ang2 = self.func_Rf_ang(point2[0],point2[1]) - 1e-3 # for ERR gradient
                    ang3 = self.func_Rf_ang(point3[0],point3[1]) + 1e-3 # for ERR gradient
                    ang4 = self.func_Rf_ang(point4[0],point4[1]) - 1e-3 # for ERR gradient
                    
                    print("Try starting points: ", point1, point2, point3, point4, ang1, ang2, ang3, ang4)
                    self.opti_solver.set_initial(self.ut1, point1[0])
                    self.opti_solver.set_initial(self.vt1, point1[1])
                    self.opti_solver.set_initial(self.ut2, point2[0])
                    self.opti_solver.set_initial(self.vt2, point2[1])
                    self.opti_solver.set_initial(self.ut3, point3[0])
                    self.opti_solver.set_initial(self.vt3, point3[1])
                    self.opti_solver.set_initial(self.ut4, point4[0])
                    self.opti_solver.set_initial(self.vt4, point4[1])
                    self.opti_solver.set_initial(self.th1, ang1)
                    self.opti_solver.set_initial(self.th2, ang2)
                    self.opti_solver.set_initial(self.th3, ang3)
                    self.opti_solver.set_initial(self.th4, ang4)
                    init_dis_open_midr = 0.5*(self.init_dis_open[0] + np.random.rand(1)[0] * (self.init_dis_open[1] - self.init_dis_open[0]))
                    # (self.init_dis_open[0] + self.init_dis_open[1])*0.5 + 2.0
                    self.opti_solver.set_initial(self.dis_open, init_dis_open_midr)
                    self.opti_solver.set_value(self.unc_thread, 0.7 * uncer_favor)
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
                            'iters': self.opti_solver.stats()['iter_count'],
                            'uncer_favor':uncer_favor,
                            'ut1':self.opti_solver.debug.value(self.ut1), 
                            'ut2':self.opti_solver.debug.value(self.ut2), 
                            'ut3':self.opti_solver.debug.value(self.ut3), 
                            'ut4':self.opti_solver.debug.value(self.ut4), 
                            'vt1':self.opti_solver.debug.value(self.vt1), 
                            'vt2':self.opti_solver.debug.value(self.vt2), 
                            'vt3':self.opti_solver.debug.value(self.vt3), 
                            'vt4':self.opti_solver.debug.value(self.vt4), 
                            'th1':self.opti_solver.debug.value(self.th1), 
                            'th2':self.opti_solver.debug.value(self.th2), 
                            'th3':self.opti_solver.debug.value(self.th3), 
                            'th4':self.opti_solver.debug.value(self.th4), 
                            'dis_open':self.opti_solver.debug.value(self.dis_open), 
                            'obj_cer':self.opti_solver.debug.value(self.obj_cer), 
                            'obj_ang':self.opti_solver.debug.value(self.obj_ang), 
                            'obj_anb':self.opti_solver.debug.value(self.obj_anb), 
                            'objfun':self.opti_solver.debug.value(self.objfun),
                        }
                    else:
                        print('\n\nSuccessing: in trial: {}\n\n'.format(trail_i))
                        solution = {
                            'state':'S',
                            'iters': self.opti_solver.stats()['iter_count'],
                            'uncer_favor':uncer_favor,
                            'ut1':sol.value(self.ut1),
                            'ut2':sol.value(self.ut2),
                            'ut3':sol.value(self.ut3),
                            'ut4':sol.value(self.ut4),
                            'vt1':sol.value(self.vt1),
                            'vt2':sol.value(self.vt2),
                            'vt3':sol.value(self.vt3),
                            'vt4':sol.value(self.vt4),
                            'th1':sol.value(self.th1),
                            'th2':sol.value(self.th2),
                            'th3':sol.value(self.th3),
                            'th4':sol.value(self.th4),
                            'dis_open':sol.value(self.dis_open),
                            'obj_cer':sol.value(self.obj_cer),
                            'obj_ang':sol.value(self.obj_ang),
                            'obj_anb':sol.value(self.obj_anb),
                            'objfun':sol.value(self.objfun),
                        }
                    # print("START DEBUG ... ")
                    # print(self.opti_solver.debug.g_describe(1))
                    # print(self.opti_solver.debug.x_describe(1))
                    # print('ang_tp2p23 :', self.opti_solver.debug.value(self.ang_tp2p23))
                    # print('ang_tp1p14 :', self.opti_solver.debug.value(self.ang_tp1p14))
                                     
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
        ut4, vt4, th4 = solution['ut4'], solution['vt4'], solution['th4']
        dis_open = solution['dis_open']

        del_open_u1 = - dis_open * np.sin(th1)
        del_open_v1 = + dis_open * np.cos(th1)
        del_open_u2 = - dis_open * np.sin(th2)
        del_open_v2 = + dis_open * np.cos(th2)
        del_open_u3 = - dis_open * np.sin(th3)
        del_open_v3 = + dis_open * np.cos(th3)
        del_open_u4 = - dis_open * np.sin(th4)
        del_open_v4 = + dis_open * np.cos(th4)
        up1 = ut1 + del_open_u1
        vp1 = vt1 + del_open_v1
        up2 = ut2 + del_open_u2
        vp2 = vt2 + del_open_v2
        up3 = ut3 + del_open_u3
        vp3 = vt3 + del_open_v3
        up4 = ut4 + del_open_u4
        vp4 = vt4 + del_open_v4

        point_center = np.array([[(up1 + up2 + up3 + up4)/4, (vp1 + vp2 + vp3 + vp4)/4]])

        p_u, p_v = point_center[0, 0], point_center[0, 1]
        if self.gripper_type == "FishGri_4F": orientation = th1 if np.cos(th1) > 0 else np.arctan(np.sin(th1)/np.cos(th1))
        if self.gripper_type == "RochuGr_4F": orientation = th1 % (np.pi/2) - np.pi/4 # for convinience
        grsap_opening = dis_open
        palm_position = self.init_palm_dists[0]/grsap_opening
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
        ut4, vt4, th4 = solution['ut4'], solution['vt4'], solution['th4']
        dis_open = solution['dis_open']

        del_open_u1 = - dis_open * np.sin(th1)
        del_open_v1 = + dis_open * np.cos(th1)
        del_open_u2 = - dis_open * np.sin(th2)
        del_open_v2 = + dis_open * np.cos(th2)
        del_open_u3 = - dis_open * np.sin(th3)
        del_open_v3 = + dis_open * np.cos(th3)
        del_open_u4 = - dis_open * np.sin(th4)
        del_open_v4 = + dis_open * np.cos(th4)
        up1 = ut1 + del_open_u1
        vp1 = vt1 + del_open_v1
        up2 = ut2 + del_open_u2
        vp2 = vt2 + del_open_v2
        up3 = ut3 + del_open_u3
        vp3 = vt3 + del_open_v3
        up4 = ut4 + del_open_u4
        vp4 = vt4 + del_open_v4

        points_palms = np.array([[up1, vp1], [up2, vp2], [up3, vp3], [up4, vp4], [up1, vp1]])
        points_tp1 = np.array([[ut1, vt1], [up1, vp1]])
        points_tp2 = np.array([[ut2, vt2], [up2, vp2]])
        points_tp3 = np.array([[ut3, vt3], [up3, vp3]])
        points_tp4 = np.array([[ut4, vt4], [up4, vp4]])

        point_center = np.array([[(up1 + up2 + up3 + up4)/4, (vp1 + vp2 + vp3 + vp4)/4]])
        sta, cer, obj = solution['state'], solution['obj_cer'], solution['objfun']
        
        # start ploting
        plt.plot(points_palms[:, 1], points_palms[:, 0], '--', linewidth=2,  color='b', zorder=0)
        plt.plot(points_tp1[:, 1], points_tp1[:, 0], linewidth=4, color='k' if sta == 'S' else 'r', zorder=5)
        plt.plot(points_tp2[:, 1], points_tp2[:, 0], linewidth=4, color='k' if sta == 'S' else 'r', zorder=5)
        plt.plot(points_tp3[:, 1], points_tp3[:, 0], linewidth=4, color='k' if sta == 'S' else 'r', zorder=5)
        plt.plot(points_tp4[:, 1], points_tp4[:, 0], linewidth=4, color='k' if sta == 'S' else 'r', zorder=5)
        plt.scatter(point_center[:, 1], point_center[:, 0], s = 10, color='w', zorder=10)
        plt.text(10, 20, '{}: {:.1f},{:.1f}'.format(sta, cer, obj), fontdict=dict(fontsize=10, color='w', weight='bold'),
            bbox=dict(fill=True, color='k', linewidth=1, alpha=0.8))
        # plt.show()
        return plt
    

# ###
# ### Try it now !
# ###
# gripper_optimizer = GraspOptimSolver_4F(gripper_type='FishGrip4F') # FishGrip4F Rochu4F
# gripper_optimizer.refresh_mapsfuns(angle_map, certainty_map)
# gripper_optimizer.init_solver()

# sol_pack, sol_final, sol_best = gripper_optimizer.iterate_solver(\
#         max_iter=500, total_intervals=3, trials_each_interval=10, rich_output=False, decrease_rate=0.8)

# # for sol_pack_i in sol_pack: gripper_optimizer.plot_result(angle_map, sol_pack_i)

# best_objs = []
# for sol_pack_i in sol_pack: 
#     best_objs.append(sol_pack_i['objfun'])
#     best_objs.append(sol_pack_i['obj_min'])
#     best_objs.append(sol_pack_i['state'])
#     print('\n\nsol_pack:', sol_pack_i)

# if(len(sol_pack) > 0): gripper_optimizer.plot_result_4F(angle_map, sol_best)
# print('\n\nsol_best:', sol_best)
# if(len(sol_pack) > 0): gripper_optimizer.plot_result_4F(angle_map, sol_final)
# print('\n\nsol_final:', sol_final)