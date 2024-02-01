from typing import Tuple

import numpy as np
import torch
from nnunet.network_architecture.generic_modular_residual_UNet import get_default_network_config
from nnunet.network_architecture.REUnet_modular import REUnet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from nnunet.network_architecture.neural_network import SegmentationNetwork


class nnUNetTrainerV2_REUNet(nnUNetTrainerV2):
    def initialize_network(self):
        if self.threeD:
            cfg = get_default_network_config(3, None, norm_type="in")
        else:
            cfg = get_default_network_config(1, None, norm_type="in")

        stage_plans = self.plans['plans_per_stage'][self.stage]
        conv_kernel_sizes = stage_plans['conv_kernel_sizes']
        blocks_per_stage_encoder = stage_plans['num_blocks_encoder']
        blocks_per_stage_decoder = stage_plans['num_blocks_decoder']
        pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        self.network = REUnet(self.num_input_channels, self.base_num_features, blocks_per_stage_encoder, 2,
                              pool_op_kernel_sizes, conv_kernel_sizes, cfg, self.num_classes,
                              blocks_per_stage_decoder, True, False, 320, InitWeights_He(1e-2))

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.validate(self, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                                     step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
                                     overwrite=overwrite, validation_folder_name=validation_folder_name,
                                     debug=debug, all_in_gpu=all_in_gpu,
                                     segmentation_export_kwargs=segmentation_export_kwargs,
                                     run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.decoder.deep_supervision = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=do_mirroring,
                                                                             mirror_axes=mirror_axes,
                                                                             use_sliding_window=use_sliding_window,
                                                                             step_size=step_size,
                                                                             use_gaussian=use_gaussian,
                                                                             pad_border_mode=pad_border_mode,
                                                                             pad_kwargs=pad_kwargs,
                                                                             all_in_gpu=all_in_gpu,
                                                                             verbose=verbose,
                                                                             mixed_precision=mixed_precision)
        self.network.decoder.deep_supervision = ds
        return ret

    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = True
        ret = nnUNetTrainer.run_training(self)
        self.network.decoder.deep_supervision = ds
        return ret
    
    
    def predict_preprocessed_data_return_seg_and_softmax_and_segment_ability_map_adj_ds(
        self, data: np.ndarray, do_mirroring: bool = True,
        mirror_axes: Tuple[int] = None,
        use_sliding_window: bool = True, step_size: float = 0.5,
        use_gaussian: bool = True, pad_border_mode: str = 'constant',
        pad_kwargs: dict = None, all_in_gpu: bool = False,
        verbose: bool = True, mixed_precision=True,
        true_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = self.predict_preprocessed_data_return_seg_and_softmax_and_segment_ability_map(
            data, do_mirroring=do_mirroring,
            mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            use_gaussian=use_gaussian,
            pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs,
            all_in_gpu=all_in_gpu,
            verbose=verbose,
            mixed_precision=mixed_precision,
            true_mask=true_mask)
        self.network.decoder.deep_supervision = ds
        return ret
    
    # for SAM (segment ability map)
    def predict_preprocessed_data_return_seg_and_softmax_and_segment_ability_map(
        self, data: np.ndarray, do_mirroring: bool = True,
        mirror_axes: Tuple[int] = None,
        use_sliding_window: bool = True, step_size: float = 0.5,
        use_gaussian: bool = True, pad_border_mode: str = 'constant',
        pad_kwargs: dict = None, all_in_gpu: bool = False,
        verbose: bool = True, mixed_precision: bool = True,
        true_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_and_segment_ability_map_3D(
            data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window, step_size=step_size,
            patch_size=self.patch_size, regions_class_order=self.regions_class_order,
            use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
            mixed_precision=mixed_precision,
            true_mask=true_mask)
        self.network.train(current_mode)
        return ret

