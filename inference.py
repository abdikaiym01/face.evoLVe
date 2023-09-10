import torch
import cv2
from backbone.model_irse import IR_18
import torchvision.transforms as transforms
import numpy as np
from util.utils import l2_norm

BEST_TRESHOLD = 1.526
device = torch.device("cuda:0")

ccrop = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

ir_18_model = IR_18([32, 32])
ir_18_model.load_state_dict(torch.load('/root/face.evoLVe/model/Backbone_IR_18_Epoch_35_Batch_16765_Time_2023-09-10-14-55_checkpoint.pth'))
ir_18_model.cuda()
ir_18_model.eval()

img1 = cv2.imread('/root/face.evoLVe/datasets/oz_test/lfw/positive/0a0a6f80-0a87-4da5-b293-470e1c4bdfe0/b62b3185-cd62-4a7d-97ff-5f19be10516c.jpg')
img2 = cv2.imread('/root/face.evoLVe/datasets/oz_test/lfw/positive/0a0a6f80-0a87-4da5-b293-470e1c4bdfe0/f0bad6e8-2562-4e6b-a964-be748742db25.jpg')

imgs = torch.stack((ccrop(img1), ccrop(img2))).cuda()

vectors = l2_norm(ir_18_model(imgs)).cpu().detach()

diff = torch.subtract(vectors[0, :], vectors[1, :])
dist = torch.sum(torch.square(diff))

if BEST_TRESHOLD < dist:
    print(f'DIST: {dist}, faces are not same')
else:
    print(f'DIST: {dist}, faces are same')