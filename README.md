# REUnet
An Externally Validated Deep Learning Model for Segmentation of Intracerebral and Intraventricular Hemorrhage on Head CT and Segmentation Quality Assessment

# Requirements
Install [`nnU-Net`](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), which is the framework for running REUnet.

Download [`StripSkullCT`](https://github.com/WuChanada/StripSkullCT) for brain extraction.

# Installation
After installing `nnU-Net`, you can install `REUnet` by this command: `python ./install_REUnet.py`.

# Getting REUnet trained models
Running `python ./download_REUnet_trained_models.py` to download REUnet trained models.

This contains 5 models produced by 5 fold cross-validation, and the size is 3.1GB.

# Usage
After installation, you can get ICH and IVH segmentation by following steps:

1. Saving your brain CT image by `Nifti` format

2. Applying brain extraction by [`StripSkullCT`](https://github.com/WuChanada/StripSkullCT) in `Matlab`. 

3. Running the following command:

 ```
ip=INPUT_FOLDER 
op=OUTPUT_FOLDER 
fold=FOLD  # choose from [0, 1, 2, 3, 4]

nnUNet_predict_sam -i ${ip} -o ${op} -t 607 -m 3d_fullres -f ${fold} -chk model_best -tr nnUNetTrainerV2_REUNet -p nnUNetPlans_REUNet_v2.1 --save_npz
```
