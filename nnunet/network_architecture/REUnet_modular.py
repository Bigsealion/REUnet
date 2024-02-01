from nnunet.network_architecture.generic_modular_UNet import PlainConvUNetDecoder
from nnunet.network_architecture.custom_modules.conv_blocks import BasicResidualBlock

from nnunet.network_architecture.generic_modular_residual_UNet import ResidualUNetEncoder

from nnunet.network_architecture.neural_network import SegmentationNetwork

from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
import numpy as np
import torch
from torch import nn
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from typing import Union, Tuple, List, Dict
from batchgenerators.augmentations.utils import pad_nd_image
# hooks to get feature maps
from nnunet.utilities.interpretability_method.get_feature_map_by_hook import REUnetHookForFeatureMaps, ProtoSegMultiClass, ProtoSegMultiClassBinary, ProtoSegMultiClassNumpyOnCPU, ProtoSegMultiClassSkipEmptyClass, ProtoSegMultiClassNumpyOnCPUSkipEmptyClass
import torch.nn.functional as F
from torch.cuda.amp import autocast
from nnunet.utilities.random_stuff import no_op
import time


# fbU multi decode layer, midified by FabiansUNet ---------------------------------------------
class REUnet(SegmentationNetwork):
    """
    Residual Encoder, Plain conv decoder
    """
    use_this_for_2D_configuration = 1244233721.0  # 1167982592.0
    use_this_for_3D_configuration = 1230348801.0
    default_blocks_per_stage_encoder = (1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4)
    default_blocks_per_stage_decoder = (1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4)  # modify
    default_min_batch_size = 2  # this is what works with the numbers above

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None,
                 block=BasicResidualBlock,
                 props_decoder=None):
        super().__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = ResidualUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                           feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                           props, default_return_skips=True, max_num_features=max_features, block=block)
        props['dropout_op_kwargs']['p'] = 0
        if props_decoder is None:
            props_decoder = props
        self.decoder = PlainConvUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props_decoder,
                                            deep_supervision, upscale_logits)
        if initializer is not None:
            self.apply(initializer)
        
        # get feautre maps
        self.feature_map_hooks = REUnetHookForFeatureMaps(self)
        
        # get SAM softmax (not used)
        # self.sam_calculator = ProtoSegMultiClass(ndims="3d")
        # self.sam_calculator = ProtoSegMultiClassBinary(ndims="3d", is_ignore_background=False)
        # self.sam_calculator = ProtoSegMultiClassNumpyOnCPU(ndims="3d")
        
        # Used SAM calculator, GPU or CPU version
        self.sam_calculator = ProtoSegMultiClassSkipEmptyClass(ndims="3d")
        # self.sam_calculator = ProtoSegMultiClassNumpyOnCPUSkipEmptyClass(ndims="3d")
        
        # In fact, only output node of Dec3 and Dec4 Layer
        self.is_get_node_sam = True
        
        # resize mask or feature map for get SAM
        # if is_sam_resize_feature_map is True, will resize feature map, else resize mask to get SAM
        self.is_sam_resize_feature_map = False

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, num_conv_per_stage_encoder,
                                        num_conv_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
        enc = ResidualUNetEncoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                  num_modalities, pool_op_kernel_sizes,
                                                                  num_conv_per_stage_encoder,
                                                                  feat_map_mul_on_downscale, batch_size)
        dec = PlainConvUNetDecoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                   num_classes, pool_op_kernel_sizes,
                                                                   num_conv_per_stage_decoder,
                                                                   feat_map_mul_on_downscale, batch_size)

        return enc + dec
    
    # modified from _internal_maybe_mirror_and_pred_3D
    # Add hook to get feature maps
    def _internal_maybe_mirror_and_pred_and_segment_ability_map_3D(
        self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
        do_mirroring: bool = True,
        mult: np.ndarray or torch.tensor = None) -> Tuple[torch.tensor, dict]:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        x = maybe_to_torch(x)
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred
                # create feature maps, because shape of fm is different, so using themselves as init
                fm_result_torch_dic = {k: 1 / num_results * v.clone() for k, v in self.feature_map_hooks.images.items()}
                if torch.cuda.is_available():
                    for fm_k, fm_v in fm_result_torch_dic.items():
                        fm_result_torch_dic[fm_k] = to_cuda(fm_v, gpu_id=self.get_device())

            if m == 1 and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, ))))
                result_torch += 1 / num_results * torch.flip(pred, (4,))
                # feature maps
                for fm_k, fm_v in self.feature_map_hooks.images.items():
                    fm_result_torch_dic[fm_k] += 1 / num_results * torch.flip(fm_v, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3,))
                # feature maps
                for fm_k, fm_v in self.feature_map_hooks.images.items():
                    fm_result_torch_dic[fm_k] += 1 / num_results * torch.flip(fm_v, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))
                # feature maps
                for fm_k, fm_v in self.feature_map_hooks.images.items():
                    fm_result_torch_dic[fm_k] += 1 / num_results * torch.flip(fm_v, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2,))
                # feature maps
                for fm_k, fm_v in self.feature_map_hooks.images.items():
                    fm_result_torch_dic[fm_k] += 1 / num_results * torch.flip(fm_v, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))
                # feature maps
                for fm_k, fm_v in self.feature_map_hooks.images.items():
                    fm_result_torch_dic[fm_k] += 1 / num_results * torch.flip(fm_v, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))
                # feature maps
                for fm_k, fm_v in self.feature_map_hooks.images.items():
                    fm_result_torch_dic[fm_k] += 1 / num_results * torch.flip(fm_v, (3, 2,))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))
                # feature maps
                for fm_k, fm_v in self.feature_map_hooks.images.items():
                    fm_result_torch_dic[fm_k] += 1 / num_results * torch.flip(fm_v, (4, 3, 2))
        
        # hook remove # DO NOT USED IN HERE!
        # self.feature_map_hooks.remove()
        # get SAM of output for test code
        fm_result_torch_dic["out"] = result_torch[:, :].clone()
        
        # resize shape of feature maps to image shape to get SAM
        # Or, resize mult to feature map size
        mult_fm_dic = dict()
        if self.is_sam_resize_feature_map:
            # in this branch, resize feature map to mask (mult) size
            for fm_k, fm_v in fm_result_torch_dic.items():
                fm_result_torch_dic[fm_k] = F.interpolate(
                    fm_v, size=pred.shape[2:], mode="trilinear", align_corners=False)
                if mult is not None:
                    fm_result_torch_dic[fm_k][:, :] *= mult
        else:
            # in this branch, resize mult to feature map size to speed up calculate
            if mult is not None:
                for fm_k, fm_v in fm_result_torch_dic.items():
                    mult_fm = F.interpolate(
                        mult.unsqueeze(0).unsqueeze(0), size=fm_v.shape[2:], mode="trilinear", align_corners=False)
                    mult_fm_dic[fm_k] = mult_fm
                    fm_result_torch_dic[fm_k] *= mult_fm
                    
        # FOR ALL NODES OF LAST 2 LAYER -------------------------
        # shape of fm_result_torch_dic["decoder_StackedConvLayers_4"] is [B, C, X, Y, Z]
        if self.is_get_node_sam:
            if "decoder_StackedConvLayers_4" in fm_result_torch_dic.keys():
                for node_i in range(fm_result_torch_dic["decoder_StackedConvLayers_4"].shape[1]):
                    fm_result_torch_dic[f"D4_node_{node_i}"] = fm_result_torch_dic[
                        "decoder_StackedConvLayers_4"][:, node_i: node_i+1, ...].clone()
                    if (mult is not None) and (not self.is_sam_resize_feature_map):
                        mult_fm_dic[f"D4_node_{node_i}"] = mult_fm_dic["decoder_StackedConvLayers_4"].clone()
            if "decoder_StackedConvLayers_3" in fm_result_torch_dic.keys():
                for node_i in range(fm_result_torch_dic["decoder_StackedConvLayers_3"].shape[1]):
                    fm_result_torch_dic[f"D3_node_{node_i}"] = fm_result_torch_dic[
                        "decoder_StackedConvLayers_3"][:, node_i: node_i+1, ...].clone()
                    if (mult is not None) and (not self.is_sam_resize_feature_map):
                        mult_fm_dic[f"D3_node_{node_i}"] = mult_fm_dic["decoder_StackedConvLayers_3"].clone()
        # FOR ALL NODES OF LAST 2 LAYER, END ------------------------

        # get SAM map
        # SAM_softmax_dic = dict()
        # patch_class = torch.argmax(result_torch, dim=1, keepdim=False)
        # patch_class_onehot = F.one_hot(patch_class, result_torch.shape[1]).permute(0, 4, 1, 2, 3)
        # for fm_k, fm_v in fm_result_torch_dic.items():
        #     SAM_softmax_dic[fm_k] = self.sam_calculator(fm_v, patch_class_onehot)
        #     if mult is not None:
        #         SAM_softmax_dic[fm_k][:, :] *= mult
                
        if mult is not None:
            result_torch[:, :] *= mult
            
        # return result_torch; SAM_softmax_dic; mult dic (resize to feature map if self.is_sam_resize_feature_map)
        return result_torch, fm_result_torch_dic, mult_fm_dic

    
    def _internal_predict_3D_3Dconv_tiled_and_segment_ability_map(
        self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
        patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
        pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
        verbose: bool,
        true_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
                if verbose: print("done")
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

            #predict on cpu if cuda not available
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())

        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            # for other dims map
            aggregated_nb_of_predictions_dim1 = np.zeros([1] + list(data.shape[1:]), dtype=np.float32)

        is_first_patch_loop = True  # for SAM
        loop_n = 0
        loop_all_n = len(steps[0])*len(steps[1])*len(steps[2])
        time_op = time.time()
        for x_n, x in enumerate(steps[0]):
            lb_x = x
            ub_x = x + patch_size[0]
            for y_n, y in enumerate(steps[1]):
                lb_y = y
                ub_y = y + patch_size[1]
                for z_n, z in enumerate(steps[2]):
                    lb_z = z
                    ub_z = z + patch_size[2]
                    
                    loop_n += 1
                    print("{}/{} start, {:.2f}s".format(loop_n, loop_all_n, time.time() - time_op))
                    
                    # patch of SAM have same process of predicted_patch
                    predicted_patch, SAM_softmax_dic, mult_fm_dic = self._internal_maybe_mirror_and_pred_and_segment_ability_map_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring,
                        gaussian_importance_map)
                    # remove first dim (batch size, and length is 1)
                    # where, shape in SAM_softmax_dic is [C, X, Y, Z]
                    predicted_patch = predicted_patch[0]  
                    SAM_softmax_dic = {k: v[0] for k, v in SAM_softmax_dic.items()}
                    if mult_fm_dic:
                        mult_fm_dic = {k: v[0].cpu().numpy() for k, v in mult_fm_dic.items()}

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                        SAM_softmax_dic = {k: v.half() for k, v in SAM_softmax_dic.items()}
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()
                        SAM_softmax_dic = {k: v.detach().cpu().numpy() for k, v in SAM_softmax_dic.items()}
                    
                    print("{}/{} add feature map, {:.2f}s".format(loop_n, loop_all_n, time.time() - time_op))
                    # add patch feature map, SAM
                    if self.is_sam_resize_feature_map:
                        # get SAM init
                        if is_first_patch_loop:
                            aggregated_SAMs_dic = {k: np.zeros([v.shape[0]] + list(data.shape[1:]), dtype=np.float32)
                                                   for k, v in SAM_softmax_dic.items()}
                            is_first_patch_loop = False
                        # set SAM patch to whole image
                        for sam_k, sam_v in SAM_softmax_dic.items(): 
                            aggregated_SAMs_dic[sam_k][:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += sam_v
                        # add nb weight
                        aggregated_nb_of_predictions_dim1[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds
                    else:
                        # get SAM init
                        if is_first_patch_loop:
                            # This Code is replaced by the following 'for loop' ----------------------
                            # get image size of each layer
                            # Get the scale of each patch
                            # patch_scale_dic = {k: [v.shape[i+1]/predicted_patch.shape[i+1]
                            #                        for i in range(len(v.shape[1:]))] 
                            #                    for k, v in SAM_softmax_dic.items()}
                            # aggregated_SAMs_shape_dic = {k: [round(v[i] * data.shape[i+1])
                            #                                  for i in range(len(v))]
                            #                              for k, v in patch_scale_dic.items()}
                            # patch_step = {k: [[round(s * v[i])
                            #                    for s in steps[i]]
                            #                   for i in range(len(v))]
                            #               for k, v in patch_scale_dic.items()}
                            # This Code is replaced by the following 'for loop' ----------------------
                            
                            SAM_patch_scale_dic = {k: []for k in SAM_softmax_dic.keys()}
                            SAM_aggregated_shape_dic = {k: []for k in SAM_softmax_dic.keys()}
                            SAM_steps_dic = {k: []for k in SAM_softmax_dic.keys()}
                            SAM_aggregated_nb_of_predictions_dic = {k: []for k in SAM_softmax_dic.keys()}
                            for sam_k, sam_v in SAM_softmax_dic.items():
                                # each dim of 3d patch
                                for dim_i in range(len(sam_v.shape[1:])):
                                    # get scale of each patch -------------------------
                                    patch_scale_i = sam_v.shape[dim_i+1] / predicted_patch.shape[dim_i+1]
                                    SAM_patch_scale_dic[sam_k].append(patch_scale_i)
                                    
                                    # get aggregated shape of each patch --------------
                                    aggregated_SAMs_shape_i = round(patch_scale_i * data.shape[dim_i+1])
                                    SAM_aggregated_shape_dic[sam_k].append(aggregated_SAMs_shape_i)
                                    
                                    # get steps of each patch --------------------------
                                    patch_step_i = [round(i * patch_scale_i) for i in steps[dim_i]]
                                    # if last patch not in edge of image, adjust it to edge
                                    patch_step_i[-1] += aggregated_SAMs_shape_i - patch_step_i[-1] - sam_v.shape[1:][dim_i]
                                    SAM_steps_dic[sam_k].append(patch_step_i)
                            
                            # get aggregated_SAMs with feature map scaled shape
                            aggregated_SAMs_dic = {k: np.zeros([SAM_softmax_dic[k].shape[0]] + v, dtype=np.float32)
                                                   for k, v in SAM_aggregated_shape_dic.items()}
                            is_first_patch_loop = False
                            
                            # init aggregated_nb for feature map
                            SAM_aggregated_nb_of_predictions_dic = {k: np.zeros([1] + v, dtype=np.float32)
                                                                    for k, v in SAM_aggregated_shape_dic.items()}
                            
                        # set SAM patch to whole image
                        for sam_k, sam_v in SAM_softmax_dic.items():
                            # shape of sam_k is [C, x, y, z]
                            sam_lb_x = SAM_steps_dic[sam_k][0][x_n]
                            sam_ub_x = sam_lb_x + sam_v.shape[1]
                            sam_lb_y = SAM_steps_dic[sam_k][1][y_n]
                            sam_ub_y = sam_lb_y + sam_v.shape[2]
                            sam_lb_z = SAM_steps_dic[sam_k][2][z_n]
                            sam_ub_z = sam_lb_z + sam_v.shape[3]

                            aggregated_SAMs_dic[sam_k][:, sam_lb_x:sam_ub_x, sam_lb_y:sam_ub_y, sam_lb_z:sam_ub_z] += sam_v
                            SAM_aggregated_nb_of_predictions_dic[sam_k][
                                :, sam_lb_x:sam_ub_x, sam_lb_y:sam_ub_y, sam_lb_z:sam_ub_z] += mult_fm_dic[sam_k]
                    
                    # true pred, add patch and nb weight
                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds
        
        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i])
             for i in range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])

        # apply slicer on true
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]
        # dim 1 in slicer is higher than aggregated_nb_of_predictions_dim1 (which is 1), so have no effect
        aggregated_nb_of_predictions_dim1 = aggregated_nb_of_predictions_dim1[slicer]
        
        # computing the class_probabilities by dividing the aggregated result with result_numsamples ---------------
        aggregated_results /= aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = aggregated_results.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = aggregated_results.detach().cpu().numpy()
            else:
                class_probabilities_here = aggregated_results
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            aggregated_results = aggregated_results.detach().cpu().numpy()

        # SAM, have same process of predicted segmentation
        predicted_SAMs_dic = dict()
        aggregated_SAMs_out_dic = dict()
        if true_mask is not None:
            true_mask = to_cuda(torch.from_numpy(true_mask), gpu_id=self.get_device()).to(torch.int64)
        if self.is_sam_resize_feature_map:
            predicted_segmentation_tensor = to_cuda(torch.from_numpy(predicted_segmentation), gpu_id=self.get_device())
            # get onehot mask to calculate SAM by feature map
            # If there is no true mask, use the predicted mask to replace it
            if true_mask is not None:
                pred_seg_onehot = F.one_hot(
                    true_mask[0], aggregated_results.shape[0]).permute(3, 0, 1, 2).unsqueeze_(0)
            else:
                pred_seg_onehot = F.one_hot(
                        predicted_segmentation_tensor, aggregated_results.shape[0]).permute(3, 0, 1, 2).unsqueeze_(0)
            if not self.sam_calculator.run_in_gpu:
                pred_seg_onehot_np = pred_seg_onehot.detach().cpu().numpy()
            
            sam_get_n = 0
            for sam_k, sam_v in aggregated_SAMs_dic.items():
                sam_get_n += 1
                print("{}/{} {} get SAM".format(sam_get_n, len(aggregated_SAMs_dic), sam_k))
                # if sam in here is feature map, using this slicer to get correct dim 0
                sam_slicer = tuple([slice(0, sam_v.shape[0])] + list(slicer[1:]))
                sam_v = sam_v[sam_slicer]

                # computing the class_probabilities by dividing the aggregated result with result_numsamples
                sam_v /= aggregated_nb_of_predictions_dim1
                
                # get SAM by feature map -----------------------------------
                if self.sam_calculator.run_in_gpu:
                    # in fact, sam_v in here is feature map but not SAM
                    # get binary SAM of each label ---------------------------
                    sam_v_tensor = to_cuda(torch.from_numpy(sam_v), gpu_id=self.get_device()).unsqueeze_(0)
                    SAM_list = self.sam_calculator(sam_v_tensor, pred_seg_onehot)
                    # # for sam_class_i in range(len(SAM_list)):
                    # #     # using [0] to remova dim batch
                    # #     aggregated_SAMs_out_dic[sam_k + f"SAM_c{sam_class_i}"] = SAM_list[sam_class_i][0].detach().cpu().numpy()
                    
                    # get an multi-label SAM ---------------------------------
                    # here, SAM_list is a softmax tensor but not list
                    # using [0] to remova dim batch
                    aggregated_SAMs_out_dic[sam_k + "_SAM"] = SAM_list[0].detach().cpu().numpy()
                else:
                    # calculate SAM on CPU ---------------------------------
                    # using 0 to remove dim 0 (batch)
                    aggregated_SAMs_out_dic[sam_k + "_SAM"] = self.sam_calculator.get_segment_abillty_map(
                        np.expand_dims(sam_v, axis=0), pred_seg_onehot_np)[0]
                
                # END get SAM by feature map ------------------------------------

                # this argmax DO NOT be used in after code in fact ...
                if regions_class_order is None:
                    sam_segmentation = sam_v.argmax(0)
                else:
                    if all_in_gpu:
                        sam_class_probabilities_here = sam_v.detach().cpu().numpy()
                    else:
                        sam_class_probabilities_here = sam_v
                    sam_segmentation = np.zeros(sam_class_probabilities_here.shape[1:], dtype=np.float32)
                    for i, c in enumerate(regions_class_order):
                        sam_segmentation[sam_class_probabilities_here[i] > 0.5] = c

                if all_in_gpu:
                    if verbose: print("SAM, copying results to CPU")

                    if regions_class_order is None:
                        sam_segmentation = sam_segmentation.detach().cpu().numpy()

                    sam_v = sam_v.detach().cpu().numpy()
                    
                # out SAM prob and pred (if is feature map, the "predicted_SAMs_dic" is meaningless)
                aggregated_SAMs_dic[sam_k] = sam_v
                predicted_SAMs_dic[sam_k] = sam_segmentation
        else:
            sam_get_n = 0
            for sam_k, sam_v in aggregated_SAMs_dic.items():
                sam_get_n += 1
                print("{}/{} {} get SAM".format(sam_get_n, len(aggregated_SAMs_dic), sam_k))
            
                # apply scaled slicer to feature map ---------------------------
                # get SAM slicer with shape of feature map
                sam_slicer = []
                # slicer_n 0, 1, 2 means dim x, y, z
                for slicer_n, slicer_i in enumerate(slicer[1:]):
                    sam_slicer.append(slice(
                        round(slicer_i.start * SAM_patch_scale_dic[sam_k][slicer_n]),
                        round(slicer_i.stop * SAM_patch_scale_dic[sam_k][slicer_n])))
                # where 'len(slicer)' is right, not bug
                sam_slicer = tuple([slice(0, aggregated_SAMs_dic[sam_k].shape[i])
                                    for i in range(len(aggregated_SAMs_dic[sam_k].shape) - (len(slicer) - 1))] + sam_slicer)
                
                # apply slicer to feature map and nb weight                
                sam_v = sam_v[sam_slicer]
                SAM_aggregated_nb_of_predictions_dic[sam_k] = SAM_aggregated_nb_of_predictions_dic[sam_k][sam_slicer]

                # computing the class_probabilities by dividing the aggregated result with result_numsamples
                sam_v /= SAM_aggregated_nb_of_predictions_dic[sam_k]
                
                # resize predict mask to feature map size -----------------------------
                predicted_segmentation_tensor = to_cuda(torch.from_numpy(predicted_segmentation), gpu_id=self.get_device())

                # get onehot predict to calculate SAM by feature map
                if true_mask is not None:
                    true_mask_resize = F.interpolate(
                        true_mask.unsqueeze(0).to(torch.float32), size=sam_v.shape[1:],
                        mode="nearest")[0, 0, ...].to(torch.int64)
                    pred_seg_onehot = F.one_hot(
                        true_mask_resize, aggregated_results.shape[0]).permute(3, 0, 1, 2).unsqueeze_(0)
                else:
                    predicted_segmentation_tensor = F.interpolate(
                        predicted_segmentation_tensor.unsqueeze(0).unsqueeze(0).to(torch.float32), size=sam_v.shape[1:],
                        mode="nearest")[0, 0, ...].to(torch.int64)
                    pred_seg_onehot = F.one_hot(
                            predicted_segmentation_tensor, aggregated_results.shape[0]).permute(3, 0, 1, 2).unsqueeze_(0)
                if not self.sam_calculator.run_in_gpu:
                    pred_seg_onehot_np = pred_seg_onehot.detach().cpu().numpy()

                # get SAM by feature map -----------------------------------
                if self.sam_calculator.run_in_gpu:
                    # in fact, sam_v in here is feature map but not SAM
                    # get binary SAM of each label ---------------------------
                    sam_v_tensor = to_cuda(torch.from_numpy(sam_v), gpu_id=self.get_device()).unsqueeze_(0)
                    # shape of SAM_list is [B, C, x, y, z]
                    SAM_list = self.sam_calculator(sam_v_tensor, pred_seg_onehot)
                    # resize SAM to image size, where [0] to remove batch dim
                    SAM_list = F.interpolate(SAM_list, size=sam_v.shape[1:], mode="nearest")

                    # get an multi-label SAM ---------------------------------
                    # here, SAM_list is a softmax tensor but not list
                    # using [0] to remova dim batch
                    aggregated_SAMs_out_dic[sam_k + "_SAM"] = SAM_list[0].detach().cpu().numpy()
                else:
                    # calculate SAM on CPU ---------------------------------
                    # using 0 to remove dim 0 (batch)
                    SAM_list = self.sam_calculator.get_segment_abillty_map(
                        np.expand_dims(sam_v, axis=0), pred_seg_onehot_np)
                    SAM_list = to_cuda(torch.from_numpy(SAM_list), gpu_id=self.get_device()).unsqueeze_(0)
                    SAM_list = F.interpolate(SAM_list, size=sam_v.shape[1:], mode="nearest")
                    aggregated_SAMs_out_dic[sam_k + "_SAM"] = SAM_list[0].detach().cpu().numpy()
                
            del SAM_aggregated_nb_of_predictions_dic  # What's the point...
                # END get SAM by feature map ------------------------------------
                
        # TEST!!!! output feature map and SAM of node in last layer -----------------------
        # aggregated_SAMs_dic = {**aggregated_SAMs_dic, **aggregated_SAMs_out_dic}
        # only output SAM but not feature map
        aggregated_SAMs_dic = aggregated_SAMs_out_dic
        # TEST END !!!! output feature map and SAM of node in last layer -------------------
        
        if verbose: print("SAM prediction done")
            
        del aggregated_nb_of_predictions


        if verbose: print("prediction done")
        
        # where, predicted_SAMs_dic is feature map, aggregated_SAMs_dic is SAM
        # SAM in aggregated_SAMs_dic has shape: [C, x, y, z], C is softmax class
        return predicted_segmentation, aggregated_results, predicted_SAMs_dic, aggregated_SAMs_dic
    
    
    # only edit patched method
    def predict_and_segment_ability_map_3D(
        self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
        use_sliding_window: bool = False,
        step_size: float = 0.5, patch_size: Tuple[int, ...] = None, 
        regions_class_order: Tuple[int, ...] = None,
        use_gaussian: bool = False, pad_border_mode: str = "constant",
        pad_kwargs: dict = None, all_in_gpu: bool = False,
        verbose: bool = True, mixed_precision: bool = True,
        true_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y, z).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :param mixed_precision: if True, will run inference in mixed precision with autocast()
        :param true_mask: if not None, will be used to calculated SAM, else calculate SAM by predicted mask
        :return:
        """
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv3d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_3Dconv_tiled_and_segment_ability_map(
                            x, step_size, do_mirroring, mirror_axes, patch_size,
                            regions_class_order, use_gaussian, pad_border_mode,
                            pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                            verbose=verbose,
                            true_mask=true_mask)
                    else:
                        res = self._internal_predict_3D_3Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs=pad_kwargs, verbose=verbose)
                elif self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_2Dconv_tiled(x, patch_size, do_mirroring, mirror_axes, step_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs, all_in_gpu, False)
                    else:
                        res = self._internal_predict_3D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res

    
