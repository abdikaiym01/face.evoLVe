import glob
import csv
import pandas as pd
import cv2
import numpy as np

def get_pairs_list(path_lfw, target):
    pairs = []
    for paths_persons in path_lfw:
        pair_paths = glob.glob(f'{paths_persons}/*')
        assert len(pair_paths) == 2
        pair = {'left': '/'.join(pair_paths[0].split('/')[-3:]), 'right': '/'.join(pair_paths[1].split('/')[-3:]), 'target': target}
        pairs.append(pair)

    return pairs

def create_pairs_csv(path_lfw = 'datasets/oz_test/lfw'):
    path_positive_subjects = glob.glob(f'{path_lfw}/positive/*')
    path_negative_subjects = glob.glob(f'{path_lfw}/negative/*')

    positive_pairs = get_pairs_list(path_positive_subjects, 1)
    negative_pairs = get_pairs_list(path_negative_subjects, 0)

    all_pos_neg_pairs = positive_pairs + negative_pairs
    keys = all_pos_neg_pairs[0].keys()
    with open(f'lfw_pos_neg_pairs.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_pos_neg_pairs)

def prepare_lfw_imgs_lfw_issame(path_csv_pair, path_lfw = 'datasets/oz_test/lfw'):
    pairs_df = pd.read_csv(path_csv_pair)
    
    img_lfw = []
    lfw_issame = []
    for _, row in pairs_df.iterrows():
        img_left = cv2.imread('/'.join([path_lfw, row['left']]), cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread('/'.join([path_lfw, row['right']]), cv2.IMREAD_GRAYSCALE)
        img_left = cv2.resize(img_left, (32, 32), interpolation = cv2.INTER_AREA)
        img_right = cv2.resize(img_right, (32, 32), interpolation = cv2.INTER_AREA)
        img_lfw.append(img_left)
        img_lfw.append(img_right)
        lfw_issame.append(row['target'])
        
    ndarray_img_lfw = np.array(img_lfw)
    
    return ndarray_img_lfw, lfw_issame


