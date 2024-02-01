# 2023.10.27 get SAM of nnUnet
import torch
import torch.nn as nn
import numpy as np


class PassLayer(nn.Module):
    # not do anything
    def __init__(self):
        super(PassLayer, self).__init__()

    def forward(self, x):
        return x


class REUnetHookForFeatureMaps(object):
    def __init__(self, net):
        self.images = {}
        
        # save hooks in this list
        self.hooks = []
        # save output feature maps in this dic
        self.images = dict()
        # Do not output encoder when QC -----------------------------------
        # for enc_n, enc_i in enumerate(net.encoder.stages):
        #     hook_name = "encoder_{}_{}".format(enc_i.__class__.__name__, enc_n)
        #     self.hooks.append(
        #         enc_i.register_forward_hook(self._build_hook(hook_name)))
        # Do not output encoder when QC -----------------------------------
        for dec_n, dec_i in enumerate(net.decoder.stages):
            if dec_n < 3: continue  # using 3 to get last 2 layer
            # if dec_n != 4: continue  # using 3 to get last 2 layer
            hook_name = "decoder_{}_{}".format(dec_i.__class__.__name__, dec_n)
            self.hooks.append(
                dec_i.register_forward_hook(self._build_hook(hook_name)))
        
        # self.hooks.append(
        #         net.decoder.segmentation_output.register_forward_hook(self._build_hook_softmax("output")))
        
        # self.nolinear = nn.Softmax(dim=1)
        self.nolinear = PassLayer()

    def _build_hook(self, idx):
        # hook must have those three input args
        def hook(module, module_input, module_output):
            self.images[idx] = module_output

        return hook
    
    def _build_hook_softmax(self, idx):
        # hook must have those three input args
        def hook(module, module_input, module_output):
            self.images[idx] = self.nolinear(module_output)

        return hook

    def remove(self):
        for hook in self.hooks:
            hook.remove()
    
# ProtoSeg ===============================================================================
# modified from https://github.com/shengfly/ProtoSeg/blob/main/ProtoSeg.py
# Expand it from binary classes to multi-classes, 2d to 3d
# but dim of input "pred" must be (B, C, W, H, ...), which is a onehot segment results
# i.e., "pred" is binary on each dim C
# class must start from 0, and continue
# class of input "pred" can be ordered by natural numbers, (e.g., 0, 1, 2, ...)
class ProtoSegMultiClass(nn.Module):
    def __init__(self, ndims='2d'):
        super().__init__()
        
        # for 1D: self.dims=(2)
        # for 2D image: self.dims=(2,3)
        # for 3D image: self.dims=(2,3,4)
        if ndims == '1d':
            self.dims = (2)
        elif ndims == '2d':
            self.dims = (2, 3)
        elif ndims == '3d':
            self.dims = (2, 3, 4)
        else:
            raise ValueError('ndims must be in [1d,2d,3d]')
            
        self.softmax = nn.Softmax(dim=1)
        self.run_in_gpu = True
        
    def forward(self, xfeat, pred, mask=None):
        #@ xfeat: the deep feature need to be inperpreted
        #@ pred: the initial segmentation results from the last layer of the network,
        #        in there, dim of pred is (B, C, H, W, ...), and must be int but not probabliity
        #@ mask is to maks out the background of the image (without any tissue)
        
        feature_seg_list = []
        # pred.shape[1] is class
        for class_i in range(pred.shape[1]):
            # using [class_i: class_i+1] as slice to keep dim of class
            # pred_i is binary in each dim C
            pred_i = pred[:, class_i: class_i+1, ...]
            if mask is not None:
                pred_i *= mask

            prototype_i = torch.sum(xfeat*pred_i, dim=self.dims, keepdim=True)
            num_prototype_i = torch.sum(pred_i, dim=self.dims, keepdim=True)
            if num_prototype_i > 0:
                prototype_i = prototype_i / num_prototype_i
        
            feature_seg_list.append(-torch.pow(xfeat-prototype_i, 2).sum(1, keepdim=True))
        
        disfeat = torch.cat(feature_seg_list, dim=1)
        pred = self.softmax(disfeat)
            
        return pred


# Fin Used
class ProtoSegMultiClassSkipEmptyClass(nn.Module):
    def __init__(self, ndims='2d'):
        super().__init__()
        
        # for 1D: self.dims=(2)
        # for 2D image: self.dims=(2,3)
        # for 3D image: self.dims=(2,3,4)
        if ndims == '1d':
            self.dims = (2)
        elif ndims == '2d':
            self.dims = (2, 3)
        elif ndims == '3d':
            self.dims = (2, 3, 4)
        else:
            raise ValueError('ndims must be in [1d,2d,3d]')
            
        self.softmax = nn.Softmax(dim=1)
        self.run_in_gpu = True
        
    def forward(self, xfeat, pred, mask=None):
        #@ xfeat: the deep feature need to be inperpreted
        #@ pred: the initial segmentation results from the last layer of the network,
        #        in there, dim of pred is (B, C, H, W, ...), and must be int but not probabliity
        #@ mask is to maks out the background of the image (without any tissue)
        
        is_skip_class = False
        used_classes_list = []
        feature_seg_list = []
        # pred.shape[1] is class
        for class_i in range(pred.shape[1]):
            # using [class_i: class_i+1] as slice to keep dim of class
            # pred_i is binary in each dim C
            pred_i = pred[:, class_i: class_i+1, ...]
            if mask is not None:
                pred_i *= mask
                
            if pred_i.max():
                prototype_i = torch.sum(xfeat*pred_i, dim=self.dims, keepdim=True)
                num_prototype_i = torch.sum(pred_i, dim=self.dims, keepdim=True)
                if num_prototype_i > 0:
                    prototype_i = prototype_i / num_prototype_i
            
                feature_seg_list.append(-torch.pow(xfeat-prototype_i, 2).sum(1, keepdim=True))
                
                used_classes_list.append(class_i)
            else:
                # if all elements in pred_i is False, skip it
                is_skip_class = True
        
        disfeat = torch.cat(feature_seg_list, dim=1)
        pred_sam = self.softmax(disfeat)
        
        # add -1 dim on softmaxed pred, which means skiped class
        if is_skip_class:
            zeros_out = torch.zeros(pred.shape) - 1
            for n, class_i in enumerate(used_classes_list):
                zeros_out[:, class_i, ...] = pred_sam[:, n, ...]
            pred_sam = zeros_out
            
        return pred_sam


# output binary SAM for each class
class ProtoSegMultiClassBinary(nn.Module):
    def __init__(self, ndims='2d', is_ignore_background=True):
        super().__init__()
        
        # for 1D: self.dims=(2)
        # for 2D image: self.dims=(2,3)
        # for 3D image: self.dims=(2,3,4)
        if ndims == '1d':
            self.dims = (2)
        elif ndims == '2d':
            self.dims = (2, 3)
        elif ndims == '3d':
            self.dims = (2, 3, 4)
        else:
            raise ValueError('ndims must be in [1d,2d,3d]')
            
        self.softmax = nn.Softmax(dim=1)
        self.is_ignore_background = is_ignore_background
        
        self.run_in_gpu = True
        
    def forward(self, xfeat, pred, mask=None):
        #@ xfeat: the deep feature need to be inperpreted
        #@ pred: the initial segmentation results from the last layer of the network,
        #        in there, dim of pred is (B, C, H, W, ...), and must be int but not probabliity
        #@ mask is to maks out the background of the image (without any tissue)
        feature_seg_list = []
        for class_i in range(pred.shape[1]):
            if self.is_ignore_background and (class_i == 0):
                continue
            
            feature_seg_list_i = []
            # using [class_i: class_i+1] as slice to keep dim of class
            # pred_i is binary in each dim C
            pred_i = pred[:, class_i: class_i+1, ...]
            if mask is not None:
                pred_i *= mask

            prototype_i = torch.sum(xfeat*pred_i, dim=self.dims, keepdim=True)
            num_prototype_i = torch.sum(pred_i, dim=self.dims, keepdim=True)
            if num_prototype_i > 0:
                prototype_i /= num_prototype_i
                
            # neg
            pred_i_neg = 1 - pred_i
            prototype_i_neg = torch.sum(xfeat*pred_i_neg, dim=self.dims, keepdim=True)
            num_prototype_i_neg = torch.sum(pred_i_neg, dim=self.dims, keepdim=True)
            if num_prototype_i_neg > 0:
                prototype_i_neg /= num_prototype_i_neg

            # SAM class i
            feature_seg_list_i.append(-torch.pow(xfeat-prototype_i_neg, 2).sum(1, keepdim=True))
            feature_seg_list_i.append(-torch.pow(xfeat-prototype_i, 2).sum(1, keepdim=True))
            
            disfeat = torch.cat(feature_seg_list_i, dim=1)
            # softmax SAM list
            feature_seg_list.append(self.softmax(disfeat))
        # order is class 0 ~ n
        return feature_seg_list


def softmax_np(x, axis=0):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


# get SAM on GPU will OOM, need to run on CPU
# input pred shape is [B, 1, X, Y, Z...], must be int with label
class ProtoSegMultiClassNumpyOnCPU:
    def __init__(self, ndims='2d'):
        # for 1D: self.dims=(2)
        # for 2D image: self.dims=(2,3)
        # for 3D image: self.dims=(2,3,4)
        if ndims == '1d':
            self.dims = (2)
        elif ndims == '2d':
            self.dims = (2, 3)
        elif ndims == '3d':
            self.dims = (2, 3, 4)
        else:
            raise ValueError('ndims must be in [1d,2d,3d]')
        
        self.run_in_gpu = False

    def get_segment_abillty_map(self, xfeat, pred, mask=None, classnum=None):
        #@ xfeat: the deep feature need to be inperpreted
        #@ pred: the initial segmentation results from the last layer of the network,
        #        in there, dim of pred is (B, 1, H, W, ...), and must be int but not probabliity
        # classnum: number of classes. if None, will auto get by unique pred
        #@ mask is to maks out the background of the image (without any tissue)
        
        if classnum is None:
            classnum_list = np.sort(np.unique(pred))
        else:
            classnum_list = range(classnum)
        
        feature_seg_list = []
        # pred.shape[1] is class
        for class_i in classnum_list:
            # using [class_i: class_i+1] as slice to keep dim of class
            # pred_i is binary in each dim C
            # pred_i = np.expand_dims(pred == class_i, axis=0)
            pred_i = pred == class_i
            if mask is not None:
                pred_i *= mask

            prototype_i = np.sum(xfeat*pred_i, axis=self.dims, keepdims=True)
            num_prototype_i = np.sum(pred_i, axis=self.dims, keepdims=True)
            if num_prototype_i > 0:
                prototype_i = prototype_i / num_prototype_i
        
            feature_seg_list.append(-np.power(xfeat-prototype_i, 2).sum(1, keepdims=True))
        
        pred_nosoftmax = np.concatenate(feature_seg_list, axis=1)
        # softmax is not required, and this func maybe not correct, maybe...
        # pred = softmax_np(pred_nosoftmax)

        return pred_nosoftmax



# get SAM on GPU will OOM, need to run on CPU
# input pred shape is [B, 1, X, Y, Z...], must be int with label
# Skip empty class
class ProtoSegMultiClassNumpyOnCPUSkipEmptyClass:
    def __init__(self, ndims='2d'):
        # for 1D: self.dims=(2)
        # for 2D image: self.dims=(2,3)
        # for 3D image: self.dims=(2,3,4)
        if ndims == '1d':
            self.dims = (2)
        elif ndims == '2d':
            self.dims = (2, 3)
        elif ndims == '3d':
            self.dims = (2, 3, 4)
        else:
            raise ValueError('ndims must be in [1d,2d,3d]')
        
        self.run_in_gpu = False

    def get_segment_abillty_map(self, xfeat, pred, mask=None):
        #@ xfeat: the deep feature need to be inperpreted
        #@ pred: the initial segmentation results from the last layer of the network,
        #        in there, dim of pred is (B, C, H, W, ...), and must be int but not probabliity
        #@ mask is to maks out the background of the image (without any tissue)
        
        is_skip_class = False
        used_classes_list = []
        feature_seg_list = []
        # pred.shape[1] is class
        for class_i in range(pred.shape[1]):
            # using [class_i: class_i+1] as slice to keep dim of class
            # pred_i is binary in each dim C
            # pred_i = np.expand_dims(pred == class_i, axis=0)
            pred_i = pred[:, class_i: class_i+1, ...]
            if mask is not None:
                pred_i *= mask
            
            # Prevent 0.0001 (or lesser) as 0
            if pred_i.max() > 0.01:
                prototype_i = np.sum(xfeat*pred_i, axis=self.dims, keepdims=True)
                num_prototype_i = np.sum(pred_i, axis=self.dims, keepdims=True)
                if num_prototype_i > 0:
                    prototype_i = prototype_i / num_prototype_i
                feature_seg_list.append(-np.power(xfeat-prototype_i, 2).sum(1, keepdims=True))
                used_classes_list.append(class_i)
            else:
                # if all elements in pred_i is False, skip it
                is_skip_class = True
                
        disfeat = np.concatenate(feature_seg_list, axis=1)
        pred_sam = softmax_np(disfeat, axis=1)
        # add -1 dim on softmaxed pred, which means skiped class
        if is_skip_class:
            zeros_out = np.zeros(pred.shape) - 1
            for n, class_i in enumerate(used_classes_list):
                zeros_out[:, class_i, ...] = pred_sam[:, n, ...]
            pred_sam = zeros_out

        return pred_sam
