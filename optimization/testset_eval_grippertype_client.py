import torch
import numpy as np

from modules.imagestr_socket import ImageStrServerClient
from modules.cnn_predictor import load_testset, param_mana
from modules.param_set import params_set


"""
Dataset test
"""
if __name__=='__main__':
    args = params_set()
    Result_dir = args.result_dir
    client = ImageStrServerClient('CLIENT', sock_id = 3456 + args.ip_bias)

    ## Handshake and Receive gripper type
    RECE_HEAD_IMG_NOMINSIZE = 8 # size: eight double values
    RECE_HEAD_STR_NOMINSIZE = 10 # size: 10 chars
    client.Send(np.ones(RECE_HEAD_IMG_NOMINSIZE).astype(np.double), 'REQUESTOPT')
    temp, gripper_id = client.Get(RECE_HEAD_IMG_NOMINSIZE * 8, RECE_HEAD_STR_NOMINSIZE, shape=(RECE_HEAD_IMG_NOMINSIZE))
    print('Received gripper_id: ', gripper_id)

    label_id_str, label_map = param_mana.analysis_labels(args.label_type)
    test_data, test_dataset = load_testset(args, label_map)

    with torch.no_grad():
        for test_epoch_ind in range(args.test_epoches):
            test_data, test_dataset = load_testset(args, label_map)
            results = {'correct': 0, 'failed': 0}
            print("\nip_bias {}, \ntest_epoch_ind {}, \nuse_rgb {}, \ntta_size {}, \nnetwork {}, \nresult_dir {}\n".format(
                args.ip_bias, test_epoch_ind, args.use_rgb, args.tta_size, args.network, args.result_dir))
            
            for idx, (x, y, didx, rot, zoom) in enumerate(test_dataset):
                input = x.cpu().numpy()
                client.Send(input, 'TestImage{}'.format(test_epoch_ind))
                grasp, strs = client.Get(5 * 8, 10, shape=(5))
                p_x, p_y, ori, wid, pal = grasp[0], grasp[1], grasp[2], grasp[3], grasp[4]
                np.set_printoptions(precision=2)
                print("Received estimate num {}/{} : ".format(idx, len(test_dataset)), grasp)

            print("Finish estimate index : ", test_epoch_ind)
