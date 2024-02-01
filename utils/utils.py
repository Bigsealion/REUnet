import numpy as np

class SegmentModeEvaluation:
    def __init__(self, classes=1, voxel_size=1, label_list=None):
        # default class number is 1
        # default mask_label is {1: label_1}, it means, in mask, 1 is label, other value is background
        if label_list:
            self.label_list = label_list
            self.classes = len(label_list)
        else:
            self.label_list = [i+1 for i in range(classes)]
            self.classes = classes

        self.voxel_size = voxel_size

    def set_voxel_size(self, voxel_size):
        self.voxel_size = voxel_size

    def set_label_list(self, label_list):
        """
        set mask label and it's name, format is {label_value: label_name}. e.g., {1: ICH, 2: IVH}
        :param label_list:  key is number
        :return:
        """
        self.label_list = label_list
        self.classes = len(label_list)

    def set_label_list_by_class_num(self, classes_num):
        """
        :param classes_num:  number
        :return:
        """
        self.label_list = [i + 1 for i in range(classes_num)]
        self.classes = classes_num

    def input_mask(self, true_mask, predict_mask):
        # Input true and predict mask (np.array)
        self.t_mask = true_mask
        self.p_mask = predict_mask

    @classmethod
    def get_eval_name(cls):
        eval_name_list = ['true_voxel_num', 'pred_voxel_num', 'true_volume', 'pred_volume', 'diff_volume',
                          'diff_precent_volume', 'dice', 'sen', 'spe', 'pre', 'jac']
        return eval_name_list

    @classmethod
    def get_eval_multi_mask_list(cls, true_mask, pred_mask, voxel_size=1, label_list=[1]):
        label_eval_dic = cls.evaluation_multi_mask(true_mask, pred_mask, voxel_size=voxel_size, label_list=label_list)
        eval_name_list = cls.get_eval_name()

        # get eval name
        label_eval_name_list = []
        label_eval_list = []
        for label_i in label_list:
            label_eval_name_list.extend([f'{label_i}_{j}' for j in eval_name_list])
            label_eval_list.extend(label_eval_dic[label_i])

        return label_eval_list, label_eval_name_list

    @classmethod
    def evaluation_multi_mask(cls, true_mask, pred_mask, voxel_size=1, label_list=[1]):
        # evaluation mult-class (contains bianry) mask
        # label_eval_dic: {1: [true_voxel_num, pred_voxel_num, ...],
        #                  2: [true_voxel_num, pred_voxel_num, ...], ...}
        label_eval_dic = {i: cls.evaluation_mask(true_mask, pred_mask, voxel_size, label=i)
                          for i in label_list}

        return label_eval_dic

    @classmethod
    def evaluation_mask(cls, true_mask, pred_mask, voxel_size=1, label=1, smooth=1e-5):
        # get volume
        true_voxel_num = cls.get_voxel_num(true_mask, label)
        pred_voxel_num = cls.get_voxel_num(pred_mask, label)
        true_volume = cls.get_volume(true_mask, voxel_size, label)
        pred_volume = cls.get_volume(pred_mask, voxel_size, label)
        diff_volume = true_volume - pred_volume
        diff_precent_volume = (diff_volume + smooth) / (true_volume + smooth)
        dice, sen, spe, pre, jac = cls.get_confusion_matrix_indicator(true_mask, pred_mask, label=label)

        eval_list = [true_voxel_num, pred_voxel_num, true_volume, pred_volume, diff_volume, diff_precent_volume,
                     dice, sen, spe, pre, jac]
        return eval_list

    @staticmethod
    def get_voxel_num(mask, label=1):
        return np.sum(mask == label)

    @staticmethod
    def get_volume(mask, voxel_size, label=1):
        return np.sum(mask == label) * voxel_size

    @staticmethod
    def get_confusion_matrix_indicator(true_mask, pred_mask, label=1, smooth=1e-5):
        TP = np.sum((true_mask == label) & (pred_mask == label))
        FP = np.sum((true_mask != label) & (pred_mask == label))
        FN = np.sum((true_mask == label) & (pred_mask != label))
        TN = np.sum((true_mask != label) & (pred_mask != label))

        dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)  # dice is f1
        sen = (TP + smooth) / (TP + FN + smooth)
        spe = (TN + smooth) / (TN + FP + smooth)
        pre = (TP + smooth) / (TP + FP + smooth)
        jac = (TP + smooth) / (TP + FP + FN + smooth)  # jac is iou; iou = dice/(2-dice)

        return dice, sen, spe, pre, jac
