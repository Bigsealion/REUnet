# 2023.11.02 calculate Dice of SAM and segments
import os
import re
import time
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
import logging
import sys
from utils import SegmentModeEvaluation


def get_logger(logging_path, logger_name='', level=logging.INFO, is_console_out=True,
               file_fmt='%(asctime)s %(name)s %(levelname)s: %(message)s',
               console_fmt='%(asctime)s %(name)s %(levelname)s: %(message)s'):
    # set logging
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create handler for writing to log file
    file_handler = logging.FileHandler(filename=logging_path, mode='w')
    file_handler.setFormatter(logging.Formatter(file_fmt))
    logger.addHandler(file_handler)

    # out to console
    if is_console_out:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(console_fmt))
        logger.addHandler(console_handler)

    return logger


# input 
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", type=str, help="Please Enter Output data dir", dest='datadir')
args = parser.parse_args()

data_dir = args.datadir

time_op = time.time()
# set parameters =========================================
img_pattern = r"(.+)\.nii\.gz$"
out_path = os.path.join(data_dir, "segment_quality_score.csv")

# if have elements, using this SAM, else using all SAM in dir
sam_name_list = []

mask_label_dic = {1: 'ICH', 2: 'IVH'}  # key is number in mask, value is meaning of label
    
# get logger =============================================
logging_path = "{}_logging.txt".format(out_path[:-4])
logger = get_logger(logging_path, "eval", is_console_out=sys.stdout.isatty())

# match SAM dir and segment ===============================
data_dir_dic = {i.name: i.path for i in os.scandir(data_dir) if i.is_dir()}

mask_dir = data_dir
logger.info("Mask dir: {}".format(mask_dir))
seg_path_dic = {re.findall(img_pattern, i.name)[0]: i.path for i in os.scandir(mask_dir)
                if (not i.is_dir()) and re.findall(img_pattern, i.name)}
matched_keys = sorted(data_dir_dic.keys() & seg_path_dic.keys())

# get label list
label_list = list(mask_label_dic.keys())
label_list.sort()

# eval info
eval_name_list = SegmentModeEvaluation.get_eval_name()
label_eval_name_list = []
for label_i in label_list:
    label_eval_name_list.extend([f'{label_i}_{j}' for j in eval_name_list])
    
# load SAM and segments ==================================
evaluation_list = []
for n, i in enumerate(matched_keys):
    # logger.info("{}/{} {}, {:.2f}s".format(n+1, len(matched_keys), i, time.time()-time_op))
    # load segment
    seg_nii = nib.load(seg_path_dic[i])
    seg_mask = seg_nii.get_fdata().round()
    
    # load sam
    if sam_name_list:
        sam_load_list = sam_name_list
    else:
        sam_load_list = os.listdir(data_dir_dic[i])
        
    for sam_n, sam_i in enumerate(sam_load_list):
        logger.info("{}/{} {}/{}, {} - {}".format(n+1, len(matched_keys), sam_n+1, len(sam_load_list),
                                                  i, sam_i))
        
        # only eval D3/4 node SAM 
        if not re.findall(r"D(3|4)_node", sam_i): 
            logger.info("\tSkip {}, not node SAM".format(sam_i))
            continue
        
        sam_nii = nib.load(os.path.join(data_dir_dic[i], sam_i))
        sam_mask = sam_nii.get_fdata().round()
        # remove index in SAM name
        sam_out_name = re.sub(f"_{i}.nii.gz", "", sam_i)
        
        voxel_size = np.abs(seg_nii.affine[0, 0] * seg_nii.affine[1, 1] * seg_nii.affine[2, 2])
        
        # calculate Dice (each classes) of Segment and SAM
        eval_dic = SegmentModeEvaluation.evaluation_multi_mask(true_mask=seg_mask, pred_mask=sam_mask,
                                                               voxel_size=voxel_size, label_list=label_list)
        
        # store results in list
        # faltten dic
        label_eval_list = []
        for label_i in label_list:
            label_eval_list.extend(eval_dic[label_i])
        # log in list
        evaluation_list.append([i, sam_out_name] + label_eval_list + [voxel_size, voxel_size])
        
        for label_i in label_list:
            logger.debug('{}: Dice={:.4f}, DiffVolumeRate={:.2f}%'.format(
                mask_label_dic[label_i], eval_dic[label_i][6], eval_dic[label_i][5]*100))
    
logger.info('{} Evaluation End {}'.format('-'*20, '-'*20))
# get evaluation df ------------------------------------------------------
evalution_df = pd.DataFrame(evaluation_list,
                            columns=tuple(['Name', "SAM"] + label_eval_name_list + ['p_voxel_size', 't_voxel_size']))

# avg to get qc ------------------------------------------------------------
qc_df = evalution_df[['Name', "1_dice", "2_dice"]]
qc_score_df = qc_df.groupby("Name").mean()
qc_score_df.rename(columns={"1_dice": "ICH_QC_score", "2_dice": "IVH_QC_score"}, inplace=True)

# save df
qc_score_df.to_csv(out_path, index=True)

logger.info('\nEvaluation CSV saved: {}'.format(out_path))


