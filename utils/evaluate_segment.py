# after predict, calculate dice of predice mask and true mask
# In mask, background is 0, ICH is 1, IVH is 2. Setting in mask_label dic

from functools import reduce
import torchio as tio
import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from utils import SegmentModeEvaluation


def match_file_mult_dir(dir_list, pattern, key_pattern=None, verbose=True):
    """
    Save matched file path across mult directory in a dic (matched by specified pattern)
    It's a extension function of 'match_source_and_mask'
    :param dir_list: list, which contains many dir (str)
    :param pattern:  str or list or tuple. Subject name pattern, which be used to match files across all dir
                     If input str, can be only 1 group in pattern, and which group will be set tot the key of match_dic
                     If using list/tuple, the content must correspond to dir_list, and have same keywords.
                     If using mult-group in pattren, please using para 'key_group_num' and 'key_pattern'
    :param key_pattern: str, default is None. Only setting when pattern is list/tuple, and have mult-group.
                        This para is pattern of key in match_dic.
                        e.g. If input pattern is: [r"HEMA_(\d{5})_(\d)", r"HEMA-(\d{5})-(\d)"]
                        Can setting 'key_pattern' to 'HEMA={}={}'
                        Using {} in key_pattern to replace each keyword
                        At this time, supposing the file names are 'HEMA_00002_1' and 'HEMA-00002-1' respectively,
                        key of match_dic (which is output) is 'HEMA=00002=1'
    :param verbose: bool, default is True. is print match info

    :return: match_dic: dic, key is file pattern, contains are file paths
    """
    print('Match pattern: {} in {} dir'.format(pattern, len(dir_list)))
    # check is str or list
    if isinstance(pattern, str):
        pattern = [pattern] * len(dir_list)
    elif not (isinstance(pattern, list) or isinstance(pattern, tuple)):
        raise TypeError('Only input str, list, or tuple! Now input type: {}'.format(type(pattern)), pattern)

    # get match dic
    # check key_group_num and key_pattern
    dir_dic_list = []
    if isinstance(key_pattern, str):
        for n, dir_i in enumerate(dir_list):
            dir_dic_list.append({key_pattern.format(*re.findall(pattern[n], i.name)[0]): i.path
                                 for i in os.scandir(dir_i)
                                 if re.search(pattern[n], i.name)})
    else:
        # get dic of each dir
        for n, dir_i in enumerate(dir_list):
            dir_dic_list.append({re.findall(pattern[n], i.name)[0]: i.path for i in os.scandir(dir_i)
                                 if re.search(pattern[n], i.name)})

    # print files in each dir
    if verbose:
        print('Number of files:')
        for n, i in enumerate(dir_dic_list):
            print('\t{}: {}'.format(dir_list[n], len(i.keys())))

    # match
    dir_pattern_list = [set(i.keys()) for i in dir_dic_list]
    match_pattern = reduce(lambda x, y: x & y, dir_pattern_list)

    # match source and mask
    match_dic = {i: list(map(lambda x: x[i], dir_dic_list)) for i in match_pattern}

    # print match info and an example
    if verbose:
        # match info
        print('Matched: {}'.format(len(match_pattern)))
        for n, pat_set_i in enumerate(dir_pattern_list):
            print('\tUnmatched in {}: {}'.format(dir_list[n], len(dir_pattern_list[n] - match_pattern)))

        # an example
        example_keys_list = list(match_dic.keys())
        example_keys_list.sort()

        if len(example_keys_list):
            example_keys = example_keys_list[0]
            print('Example:\n\tkey:\n\t\t{}\n\titems:'.format(example_keys))
            for ex_i in match_dic[example_keys]:
                print('\t\t{}'.format(ex_i))

    return match_dic


def mkdir_all(dir_all, verbose=True):
    """
    mkdir by check, and can input dir list
    :param dir_all: can input str, list or tuple. (list and tuple must store the dir name)
    :param verbose: bool, is print mkdir info
    :return: No, only mkdir
    """
    # check is str or list
    if isinstance(dir_all, str):
        dir_all = [dir_all]
    elif not (isinstance(dir_all, list) or isinstance(dir_all, tuple)):
        raise TypeError('Only input str, list, and tuple! Now input is {}'.format(type(dir_all)), dir_all)

    # run mkdir
    for mkdir_i in dir_all:
        if not os.path.exists(mkdir_i):
            os.makedirs(mkdir_i)
            if verbose:
                print('mkdir: {}'.format(mkdir_i))

    return


def main():
    time_op = time.time()
    # set parameter
    true_mask_dir = 'TRUE_MASK_DIR'  # ------------------> must modified by user

    pred_mask_dir = 'PREDICT_MASK_DIR'  # ------------------> must modified by user
    log_out_dir = 'LOG_OUT_DIR'  # ------------------> must modified by user

    # other parameters -------------------------------------------------------------------------------------------
    log_out_name = 'eval_log.csv'

    # pattern = r'(?:UNE|EXP)\d{5}.*\.nii\.gz'
    pattern = r'.*\.nii\.gz'

    mask_label_dic = {1: 'ICH', 2: 'IVH'}  # key is number in mask, value is meaning of label

    is_binary_res = False  # convert all labels to 1 (means mask will be convert to binary to get indicators)
    is_print_volume_eval = False

    # mkdir out dir ==============================================================================================
    mkdir_all(log_out_dir)

    # match
    match_dic = match_file_mult_dir([pred_mask_dir, true_mask_dir], pattern)
    subj_list = list(match_dic.keys())
    subj_list.sort()
    subj_num = len(subj_list)

    # get tio subj dataset
    tio_subj_list = []
    for subj_i in subj_list:
        # load pred and true mask
        tio_subj_list.append(tio.Subject(
            predmask=tio.LabelMap(match_dic[subj_i][0]),
            truemask=tio.LabelMap(match_dic[subj_i][1]),
            name=subj_i,
        ))
    subjects_dataset = tio.SubjectsDataset(tio_subj_list)

    # calculate dice and diff of volume -----------------------------------
    print('{} Calculate Evalutation {}'.format('-'*20, '-'*20))

    # get label list
    label_list = list(mask_label_dic.keys())
    label_list.sort()

    # set save name list
    eval_name_list = SegmentModeEvaluation.get_eval_name()
    label_eval_name_list = []
    for label_i in label_list:
        label_eval_name_list.extend([f'{label_i}_{j}' for j in eval_name_list])

    evaluation_list = []

    # run ---------------------------------------------------------------------------------------
    for n, subj_i in enumerate(subjects_dataset):

        # load pred and true mask, and volume
        name = subj_i['name']
        pred_mask_tio = subj_i['predmask']
        true_mask_tio = subj_i['truemask']

        pred_mask = pred_mask_tio.numpy()
        true_mask = true_mask_tio.numpy()

        # convert to binary mask
        if is_binary_res:
            pred_mask[pred_mask != 0] = 1
            true_mask[true_mask != 0] = 1

        pred_voxel_size = pred_mask_tio.spacing[0] * pred_mask_tio.spacing[1] * pred_mask_tio.spacing[2]
        true_voxel_size = true_mask_tio.spacing[0] * true_mask_tio.spacing[1] * true_mask_tio.spacing[2]

        # calculate evaluation
        eval_dic = SegmentModeEvaluation.evaluation_multi_mask(true_mask=true_mask, pred_mask=pred_mask,
                                                               voxel_size=pred_voxel_size, label_list=label_list)

        # print result
        # loc: dice: 6, volume diff rate: 5
        for label_i in label_list:
            print('\t{}: Dice={:.4f}, DiffVolumeRate={:.2f}%'.format(mask_label_dic[label_i],
                                                                     eval_dic[label_i][6], eval_dic[label_i][5]*100))

        # store results in list
        # faltten dic
        label_eval_list = []
        for label_i in label_list:
            label_eval_list.extend(eval_dic[label_i])
        # log in list
        evaluation_list.append([name] + label_eval_list + [pred_voxel_size, true_voxel_size])

        print('{}/{} {} {:.2f}s'.format(n+1, subj_num, name, time.time()-time_op))

    print('{} Evaluation End {}'.format('-'*20, '-'*20))

    # get evaluation df ------------------------------------------------------
    evalution_df = pd.DataFrame(evaluation_list,
                                columns=tuple(['Name'] + label_eval_name_list + ['p_voxel_size', 't_voxel_size']))

    print('\nMean:')
    for label_i in label_list:
        print('\t{}: Dice={:.4f}, DiffVolumeRate={:.2f}%'.format(
            mask_label_dic[label_i],
            np.mean(evalution_df[f'{label_i}_dice']),
            100*np.mean(evalution_df[f'{label_i}_diff_precent_volume'])))

    # save df
    log_out_path = os.path.join(log_out_dir, log_out_name)
    evalution_df.to_csv(log_out_path, index=False)

    print('\nEvaluation CSV saved: {}'.format(log_out_path))

    # special for IVH segment
    print('')
    print('Dice IVH: {}'.format(np.mean(evalution_df.loc[evalution_df['2_true_voxel_num'] != 0, '2_dice'])))
    print('IVH FP: {}%'.format(
        100 * np.sum(
            (evalution_df['2_true_voxel_num'] == 0) & (evalution_df['2_dice'] != 1)) / np.sum(
            (evalution_df['2_true_voxel_num'] == 0))))
    print('Only IVH DifVolRate: {}%'.format(
        100*(np.mean(np.abs(evalution_df.loc[evalution_df['2_true_voxel_num'] != 0, '2_diff_precent_volume'])))))

    # print grep number
    if is_print_volume_eval:
        df = evalution_df
        grep_number = [1000, 5000]
        for i in range(len(grep_number) + 1):
            if i < len(grep_number):
                vol_base = 0 if i == 0 else grep_number[i - 1]

                vol = grep_number[i]  # current volume
                ich_filter_index = (vol_base < df['1_true_volume']) & (df['1_true_volume'] <= vol)
                ivh_filter_index = ((vol_base < df['2_true_volume']) & (df['2_true_volume'] <= vol) &
                                    (df['2_true_voxel_num'] != 0))

                print('True Volume in ({}, {}] ml'.format(vol_base, vol))
                print('\tICH:')
                print('\t\tn={}'.format(np.sum(ich_filter_index)))
                print('\t\tDice: {}'.format(np.mean(df.loc[ich_filter_index, '1_dice'])))
                print('\t\tVDR: {}%'.format(100*np.mean(np.abs(df.loc[ich_filter_index, '1_diff_precent_volume']))))
                print('\tIVH:')
                print('\t\tn={}'.format(np.sum(ivh_filter_index)))
                print('\t\tDice: {}'.format(np.mean(df.loc[ivh_filter_index, '2_dice'])))
                print('\t\tVDR: {}'.format(
                    100*np.mean(df.loc[ivh_filter_index, '2_diff_precent_volume'])))
            else:
                vol = grep_number[-1]  # current volume
                ich_filter_index = (df['1_true_volume'] > vol)
                ivh_filter_index = ((df['2_true_volume'] > vol) & (df['2_true_voxel_num'] != 0))

                print('True Volume > {}ml'.format(vol))
                print('\tICH:')
                print('\t\tn={}'.format(np.sum(ich_filter_index)))
                print('\t\tDice: {}'.format(np.mean(df.loc[ich_filter_index, '1_dice'])))
                print('\t\tVDR: {}%'.format(100 * np.mean(np.abs(df.loc[ich_filter_index, '1_diff_precent_volume']))))
                print('\tIVH:')
                print('\t\tn={}'.format(np.sum(ivh_filter_index)))
                print('\t\tDice: {}'.format(np.mean(df.loc[ivh_filter_index, '2_dice'])))
                print('\t\tVDR: {}'.format(100 * np.mean(df.loc[ivh_filter_index, '2_diff_precent_volume'])))

        # warning
        if is_binary_res:
            warnings.warn('is_binary_res=True, All mask has been convert to binary! Please make sure this is correct!')


if __name__ == '__main__':
    main()
