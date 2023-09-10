import torch
import cv2
from backbone.model_irse import IR_18
import torchvision.transforms as transforms
import numpy as np
from util.utils import l2_norm
from util.utils import perform_val
import cv2
import torchvision.transforms as T
from util.prepare_lfw import prepare_lfw_imgs_lfw_issame, create_pairs_csv


ir_18_model = IR_18([32, 32])
ir_18_model.load_state_dict(torch.load('/root/face.evoLVe/model/Backbone_IR_18_Epoch_35_Batch_16765_Time_2023-09-10-14-55_checkpoint.pth'))
ir_18_model.cuda()
ir_18_model.eval()

create_pairs_csv()
lfw, lfw_issame = prepare_lfw_imgs_lfw_issame('/root/face.evoLVe/lfw_pos_neg_pairs.csv')

accuracy_lfw, best_threshold_lfw, roc_curve_lfw, far_frr_curve_tensor, eer = perform_val(False, torch.device("cuda:0"), 512, 1024, ir_18_model, lfw, lfw_issame, tta=False)

T.ToPILImage()(roc_curve_lfw).save('roc_curve.jpg')
T.ToPILImage()(far_frr_curve_tensor).save('far_frr_curve.jpg')
print(f'ACCURACY : {accuracy_lfw}, BEST_THRESHOLD : {best_threshold_lfw}, ERR : {eer}')
