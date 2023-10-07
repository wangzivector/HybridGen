import os
import logging, pickle
import matplotlib.pyplot as plt
import numpy as np
from modules.cnn_predictor import CNNPredictor
from modules.imagestr_socket import ImageStrServerClient
from modules.gripper_optimizer import GripperOptimizer
from modules.param_set import params_set


def clear_folder(Result_dir):
    """
    MiSC function
    """
    import shutil, os
    if os.path.isdir(Result_dir): shutil.rmtree(Result_dir)
    os.makedirs(Result_dir)
    os.makedirs(os.path.join(Result_dir, 'snaps'))

###
### Main function
###
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('Start workflow of grasp prediction')

    args = params_set()
    Result_dir = args.result_dir

    logging.info('Start ImageServer ...')
    comserver = ImageStrServerClient('SERVER', sock_id = 3456 + args.ip_bias)

    logging.info("Wait to receive socket client msg...")
    RECE_HEAD_IMG_NOMINSIZE = 8 # size: eight double values
    RECE_HEAD_STR_NOMINSIZE = 10 # size: 10 chars
    image, string = comserver.Get(RECE_HEAD_IMG_NOMINSIZE * 8, RECE_HEAD_STR_NOMINSIZE, shape=(RECE_HEAD_IMG_NOMINSIZE))
    while(string != 'REQUESTOPT'): 
        image, string = comserver.Get(RECE_HEAD_IMG_NOMINSIZE * 8, RECE_HEAD_STR_NOMINSIZE, shape=(RECE_HEAD_IMG_NOMINSIZE))
        logging.info("ReWait to rece socket client msg...")
    comserver.Send(np.ones(RECE_HEAD_IMG_NOMINSIZE).astype(np.double), args.gripper_type)
    
    logging.info('Start Optimizer ... . G: {}'.format(args.gripper_category))
    Optimizater = GripperOptimizer(args.gripper_category)
    gripper_optimizer = Optimizater(solver_name="ipopt", gripper_type=args.gripper_type)

    logging.info('Start CNNPredictor ...')
    predictor = CNNPredictor(args.network, args.input_channel, 
        args.label_type, args.tta_size)
    
    logging.info('clean folder: {}'.format(Result_dir))
    clear_folder(Result_dir)

    # time.sleep(1)
    eval_indx, test_epoches = 0, '0'
    while True:
###
### Start eatimate process
###
        logging.info('Waiting image ...')
        image, string = comserver.Get(720000 * args.receive_channel, 10) # 300 * 300 * 8
        test_epoches = string[-1]

        logging.info('Inferencing features ...')
        input = predictor.input_process(image, args.mode == 'dataset', chann = args.input_channel)
        certainty_map, angle_map = predictor.inference(input)
        
        logging.info('Initializing optimizer ...')
        gripper_optimizer.refresh_mapsfuns(angle_map, certainty_map)
        gripper_optimizer.init_solver() # Try to put this before while loop
        logging.info('Iterating optimizer ...')
        sol_pack, sol_best = gripper_optimizer.iterate_solver(rich_output=False)
###
### Send Optimized grasp
###
        logging.info('Getting optimized result ...')
        # print("Solution packages: ", sol_pack)
        print("Solution Best: ", sol_best)
        grasp_params, char_response = gripper_optimizer.build_grasp(sol_best)
        # grasp_params := [p_u, p_v, orientation, grsap_opening, palm_position]
        comserver.Send(np.array(grasp_params), char_response)
###
### Saving vis outputs: optimized results
###
        rgb_image = np.transpose(input[1:] - input[1:].min(), (1, 2, 0)) 
        result_vis_plt = gripper_optimizer.plot_result(certainty_map, sol_best, rgb_image)
        result_vis_plt.savefig(os.path.join(Result_dir, 'snaps', 'aOptimizor{}_{}_{}_{:.1f}.png'
            .format(test_epoches, eval_indx, sol_best['state'], sol_best['objfun'])))
        result_vis_plt.savefig(os.path.join(Result_dir, '..', 'Optimizor.png'))
###
### Saving vis outputs: certainty, angle, original
###
        logging.info('Visualizing CNN feature outputs ...')
        showlist = [certainty_map, angle_map, input[0], input[-1]]
        showname = ['cer', 'ang', 'dep', 'col']
        for i in range(len(showlist)):
            plt.figure(showname[i])
            plot = plt.matshow(showlist[i], cmap='jet')
            plt.colorbar(plot)
            plt.savefig(os.path.join(Result_dir, '..', 'Feat_{}_{}.png'.format(test_epoches, showname[i])))
            plt.savefig(os.path.join(Result_dir, 'snaps', 'Feat_{}_{}_{}.png'.format(test_epoches, eval_indx, showname[i])))
            # if sol_best['state'] == 'F':
            #     plt.savefig(os.path.join(Result_dir, 'snaps', 'NoOptiFeat{}_{}_{}.png'.format(test_epoches, eval_indx, showname[i])))
        eval_indx += 1

###
### Visualize probability and angle with arrow map
###
        # Save original image
        rgb_image = np.transpose(input[1:] - input[1:].min(), (1, 2, 0)) 
        # rgb_image = rgb_image[:, :, [2, 1, 0]] * 1.60 # augment brightness
        plt.figure("Original")
        plt.imshow(rgb_image)
        plt.savefig(os.path.join(Result_dir, '..', 'Original_{}.png'.format(test_epoches)))
        plt.savefig(os.path.join(Result_dir, 'snaps', 'Original_{}_{}_{}.png'.format(test_epoches, eval_indx, showname[i])))
        
        # Draw angle vector 
        plt.figure("qwa_map")
        plt.imshow(rgb_image)
        ts_map = params_set.qaw_to_tipspace(certainty_map*1.2, angle_map, certainty_map*10, density=5)
        for wiping, trans in ts_map:
            if trans < 0.02:
                continue
            x, y = wiping[0][1], wiping[0][0]
            dx, dy = wiping[1][1] - wiping[0][1], wiping[1][0] - wiping[0][0]
            colors = ['b', 'r', 'm', 'k']
            color = np.random.choice(colors) # choose one of the rotations in array
            size_arrow = 0.4
            plt.arrow(x, y, dx, dy, width=size_arrow, head_length=size_arrow*10, 
                        head_width = size_arrow*10, color=color, alpha = trans)
            # plt.plot(wiping[:, 1], wiping[:, 0], linewidth=1, alpha = trans)
        # plt.show()
        plt.savefig(os.path.join(Result_dir, '..', 'qwa_map_{}.png'.format(test_epoches)))
        plt.savefig(os.path.join(Result_dir, 'snaps', 'qwa_map_{}_{}_{}.png'.format(test_epoches, eval_indx, showname[i])))

        plt.close('all')

###
### Saving raw outputs: certainty, angle, original, and sol_best
###
        logging.info('Saving CNN feature outputs ...')
        savelist = np.concatenate((image, np.array([certainty_map, angle_map])), axis=0).astype(np.single)
        np.save(os.path.join(Result_dir, 'savelist_img_cer_ang_{}'.format(eval_indx)), savelist)
        ## to load with numpy:
        # image_base = np.load(im_dir)
        # print(image_base.shape)
        # image_show = np.transpose(image_base[1:4], (1, 2, 0))
        
        with open(os.path.join(Result_dir, 'best_solution_{}.txt'.format(eval_indx)), 'wb') as f: pickle.dump(sol_best,f)
        ## to read:  in Python 3: 
        # import pick
        # with open('best_solution_1.txt', 'rb') as f: d = pickle.load(f)

        # np.save(os.path.join(Result_dir, 'certainty_map_{}'.format(eval_indx)), certainty_map)
        # np.save(os.path.join(Result_dir, 'angle_map_{}'.format(eval_indx)), angle_map)