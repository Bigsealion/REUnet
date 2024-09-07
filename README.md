# REUnet
An Externally Validated Deep Learning Model for Segmentation of Intracerebral and Intraventricular Hemorrhage on Head CT and Segmentation Quality Assessment

# Requirements
## REUnet
REUnet relies on version 1.7.0 of [`nnUNet`](https://github.com/MIC-DKFZ/nnUNet/tree/v1.7.0), which is the framework for running REUnet.

To install nnUNet version 1.7.0, use the following command to download nnUNet v1.7.0:

`git clone --branch v1.7.0 --single-branch git@github.com:MIC-DKFZ/nnUNet.git`

It is necessary to modify the default `setup.py` file of nnUnet. You can replace it by the provided setup file located at `REUnet/utils/nnU_setup_example/setup.py`.

After that, you can follow [`nnUnet's instructions`](https://github.com/MIC-DKFZ/nnUNet/tree/v1.7.0) to install nnUNet.

Note that the environment variables are required by nnUNet. You can refer to the setup instructions in the following [link](https://github.com/MIC-DKFZ/nnUNet/blob/v1.7.0/documentation/setting_up_paths.md).

## StripSkullCT
Download [`StripSkullCT`](https://github.com/WuChanada/StripSkullCT) for brain extraction.

## Others
You also need to install packages like `sklearn`, `matplotlib`, `tqdm`, etc.
You can install those by `pip`.

# Installation REUnet
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
ip=INPUT_FOLDER  # ./example_data/image_be is the INPUT_FOLDER for the example data
op=OUTPUT_FOLDER  # User-defined path
fold=FOLD  # choose from [0, 1, 2, 3, 4]

# nnUNet_predict_sam is a custom command, which will output Segmentability Ability Maps for quality assessment 
nnUNet_predict_sam -i ${ip} -o ${op} -t 607 -m 3d_fullres -f ${fold} -chk model_best -tr nnUNetTrainerV2_REUNet -p nnUNetPlans_REUNet_v2.1 --save_npz
```

# Segmentation Quality Assessment
After get segmentation results, you can run following command to get quality assessment score:
```
# where ${op} is OUTPUT_FOLDER from above
python ./utils/segmentation_qc.py -d ${op}
```
A file called `segment_quality_score.csv` will be generated in the `${op}`, which records the segment quality assessment scores.

