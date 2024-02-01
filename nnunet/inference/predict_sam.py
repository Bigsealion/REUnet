import argparse
from copy import deepcopy
from typing import Tuple, Union, List

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti
from nnunet.inference.segmentation_export_sam import save_feature_map_nifti
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Process, Queue
import torch
import SimpleITK as sitk
import shutil
from multiprocessing import Pool
from nnunet.postprocessing.connected_components import load_remove_save, load_postprocessing
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.one_hot_encoding import to_one_hot
import torch.nn.functional as F
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from nnunet.inference.predict import check_input_folder_and_return_caseIDs, preprocess_multithreaded


# output segment ability map and final predict 
def predict_cases_and_segment_ability_map(
    model, list_of_lists, output_filenames, folds, save_npz, num_threads_preprocessing,
    num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True, mixed_precision=True,
    overwrite_existing=False,
    all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
    segmentation_export_kwargs: dict = None, disable_postprocessing: bool = False,
    mask_dir: str = None):
    """
    :param segmentation_export_kwargs:
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param list_of_lists: [[case0_0000.nii.gz, case0_0001.nii.gz], [case1_0000.nii.gz, case1_0001.nii.gz], ...]
    :param output_filenames: [output_file_case0.nii.gz, output_file_case1.nii.gz, ...]
    :param folds: default: (0, 1, 2, 3, 4) (but can also be 'all' or a subset of the five folds, for example use (0, )
    for using only fold_0
    :param save_npz: default: False
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param segs_from_prev_stage:
    :param do_tta: default: True, can be set to False for a 8x speedup at the cost of a reduced segmentation quality
    :param overwrite_existing: default: True
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :param mask_dir: for SAM calculate. if None, will calcualte SAM by predicted mask (self-evaluation)
                                        else, SAM will be calculated by true mask
    :return:
    """
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        # if save_npz=True then we should also check for missing npz files
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not isfile(j)) or (save_npz and not isfile(j[:-7] + '.npz'))]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)

    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)
    print("starting prediction...")
    all_output_files = []
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        all_output_files.append(all_output_files)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data
            
        # load true mask, and preprocessing
        # names of mask and output mask is same
        if mask_dir:
            true_mask_path = join(mask_dir, os.path.basename(output_filename))
            true_mask = sitk.GetArrayFromImage(sitk.ReadImage(true_mask_path))
            # crop
            bb = dct["crop_bbox"]
            true_mask = true_mask[(slice(bb[0][0], bb[0][1]), slice(bb[1][0], bb[1][1]), slice(bb[2][0], bb[2][1]))]
            # resize
            true_mask_tensor = to_cuda(torch.from_numpy(true_mask), gpu_id=trainer.network.get_device()).to(torch.float32)
            # d.shape is [C, x, y, z]
            # using [0] to remove batch dim, make true_mask_tensor.shape is also [C, x, y, z]
            true_mask = F.interpolate(true_mask_tensor.unsqueeze(0).unsqueeze(0),
                                      size=d.shape[1:], mode="nearest")[0].detach().cpu().numpy()
        else:
            true_mask = None

        print("predicting", output_filename)
        trainer.load_checkpoint_ram(params[0], False)
        softmax_all = trainer.predict_preprocessed_data_return_seg_and_softmax_and_segment_ability_map_adj_ds(
            d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision,
            true_mask=true_mask)
        
        softmax = softmax_all[1]
        sam_softmax_dic = softmax_all[3]

        for p in params[1:]:
            trainer.load_checkpoint_ram(p, False)
            softmax_all = trainer.predict_preprocessed_data_return_seg_and_softmax_and_segment_ability_map_adj_ds(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision,
                true_mask=true_mask)[1]
            
            softmax += softmax_all[1]
            for sam_k in sam_softmax_dic.keys():
                sam_softmax_dic[sam_k] += softmax_all[3][sam_k]

        if len(params) > 1:
            softmax /= len(params)
            for sam_k in sam_softmax_dic.keys():
                sam_softmax_dic[sam_k] /= len(params)

        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])
            for sam_k in sam_softmax_dic.keys():
                sam_softmax_dic[sam_k] = sam_softmax_dic[sam_k].transpose(
                    [0] + [i + 1 for i in transpose_backward])

        if save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        """There is a problem with python process communication that prevents us from communicating objects 
        larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
        communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
        enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
        patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
        then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
        filename or np.ndarray and will handle this automatically"""
        bytes_per_voxel = 4
        if all_in_gpu:
            bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
        if np.prod(softmax.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
            print(
                "This output is too large for python process-process communication. Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", softmax)
            softmax = output_filename[:-7] + ".npy"

        results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                          ((softmax, output_filename, dct, interpolation_order, region_class_order,
                                            None, None,
                                            npz_file, None, force_separate_z, interpolation_order_z),)
                                          ))
        
        # SAM ---------------------------------------------------
        # save all SAMs in sub dir
        out_dir = os.path.split(output_filename)[0]
        out_basename = os.path.split(output_filename)[1].split(".")[0]
        sub_out_dir = os.path.join(out_dir, out_basename)
        if not os.path.exists(sub_out_dir):
            os.makedirs(sub_out_dir)
        # output SAM
        sam_results_dic = {k: [] for k in sam_softmax_dic.keys()}
        # fm_results_dic = {k: [] for k in sam_softmax_dic.keys()}
        for sam_k in sam_softmax_dic.keys():
            sam_output_filename = os.path.join(
                sub_out_dir, "{}_{}.nii.gz".format(sam_k, out_basename))
            if "SAM" in sam_k:
                # sam_results_dic[sam_k].append(
                #     pool.starmap_async(
                #         save_segmentation_nifti_from_softmax,
                #         ((sam_softmax_dic[sam_k], sam_output_filename, dct, interpolation_order, region_class_order,
                #         None, None,
                #         npz_file, None, force_separate_z, interpolation_order_z),)
                #         ))
                
                # try to remove pool to prevent unexpected stops
                save_segmentation_nifti_from_softmax(
                    sam_softmax_dic[sam_k], sam_output_filename, dct, interpolation_order, region_class_order,
                    None, None,
                    npz_file, None, force_separate_z, interpolation_order_z)
            elif "decoder_StackedConvLayers" not in sam_k:
                # do not save all feature map of a layer
                save_feature_map_nifti(
                    sam_softmax_dic[sam_k], sam_output_filename, dct, interpolation_order, region_class_order,
                    None, None,
                    npz_file, None, force_separate_z, interpolation_order_z)

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]
    for sam_k in sam_results_dic.keys():
        _ = [i.get() for i in sam_results_dic[sam_k]]
        
    # There is no postprocessing in my used net, so do not apply SAM about in following
        
    # now apply postprocessing
    # first load the postprocessing properties if they are present. Else raise a well visible warning
    if not disable_postprocessing:
        results = []
        pp_file = join(model, "postprocessing.json")
        if isfile(pp_file):
            print("postprocessing...")
            shutil.copy(pp_file, os.path.abspath(os.path.dirname(output_filenames[0])))
            # for_which_classes stores for which of the classes everything but the largest connected component needs to be
            # removed
            for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
            results.append(pool.starmap_async(load_remove_save,
                                              zip(output_filenames, output_filenames,
                                                  [for_which_classes] * len(output_filenames),
                                                  [min_valid_obj_size] * len(output_filenames))))
            _ = [i.get() for i in results]
        else:
            print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                  "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
                  "%s" % model)

    pool.close()
    pool.join()



def predict_and_segment_ability_map_from_folder(
    model: str, input_folder: str, output_folder: str, folds: Union[Tuple[int], List[int]],
    save_npz: bool, num_threads_preprocessing: int, num_threads_nifti_save: int,
    lowres_segmentations: Union[str, None],
    part_id: int, num_parts: int, tta: bool, mixed_precision: bool = True,
    overwrite_existing: bool = True, mode: str = 'normal', overwrite_all_in_gpu: bool = None,
    step_size: float = 0.5, checkpoint_name: str = "model_final_checkpoint",
    segmentation_export_kwargs: dict = None, disable_postprocessing: bool = False,
    mask_dir: str = None):
    """
        here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases
        only using mode == "normal"

    :param model:
    :param input_folder:
    :param output_folder:
    :param folds:
    :param save_npz:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_segmentations:
    :param part_id:
    :param num_parts:
    :param tta:
    :param mixed_precision:
    :param overwrite_existing: if not None then it will be overwritten with whatever is in there. None is default (no overwrite)
    :return:
    """
    maybe_mkdir_p(output_folder)
    shutil.copy(join(model, 'plans.pkl'), output_folder)

    assert isfile(join(model, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(join(model, "plans.pkl"))['num_modalities']

    # check input folder integrity
    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]
    
    # match image and mask (if used)
    # Here only checking if all the masks are exist
    if mask_dir:
        # [part_id::num_parts] is same as nnUnet
        # name of mask must same as output
        true_mask_path_list = [join(mask_dir, i + ".nii.gz") for i in case_ids][part_id::num_parts]
        mask_exist_checking = all([os.path.exists(i) for i in true_mask_path_list])
        assert mask_exist_checking, "Not all images can find the corresponding mask in the mask_dir!"

    if lowres_segmentations is not None:
        assert isdir(lowres_segmentations), "if lowres_segmentations is not None then it must point to a directory"
        lowres_segmentations = [join(lowres_segmentations, i + ".nii.gz") for i in case_ids]
        assert all([isfile(i) for i in lowres_segmentations]), "not all lowres_segmentations files are present. " \
                                                               "(I was searching for case_id.nii.gz in that folder)"
        lowres_segmentations = lowres_segmentations[part_id::num_parts]
    else:
        lowres_segmentations = None

    if mode == "normal":
        if overwrite_all_in_gpu is None:
            all_in_gpu = False
        else:
            all_in_gpu = overwrite_all_in_gpu

        return predict_cases_and_segment_ability_map(
            model, list_of_lists[part_id::num_parts], output_files[part_id::num_parts], folds,
            save_npz, num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations, tta,
            mixed_precision=mixed_precision, overwrite_existing=overwrite_existing,
            all_in_gpu=all_in_gpu,
            step_size=step_size, checkpoint_name=checkpoint_name,
            segmentation_export_kwargs=segmentation_export_kwargs,
            disable_postprocessing=disable_postprocessing,
            mask_dir=mask_dir)
    else:
        raise ValueError("unrecognized mode. Must be normal when output SAM")

