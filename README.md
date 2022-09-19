# REUnet
REUnet, an externally validated deep learning model for fine segmentation of intracerebral and intraventricular hemorrhage on head CT images 

# Requirements
Install [`nnU-Net`](https://github.com/MIC-DKFZ/nnUNet), which is the framework for running REUnet.

Download [`StripSkullCT`](https://github.com/WuChanada/StripSkullCT) for brain extraction.

# Installation
After installing `nnU-Net`, you can install `REunet` by this command: `python ./REU_install`.

# load REUnet trained models
run `python ./download_REUnet_trained_models.py`

# Usage
After installation, you can get ICH and IVH segmentation by following steps:

1. Saving your brain CT image by `Nifti` format

2. Applying brain extraction by [`StripSkullCT`](https://github.com/WuChanada/StripSkullCT) in `Matlab`. 

3. Running the following command:

 ```
ip=INPUT_FOLDER 
op=OUTPUT_FOLDER 

nnUNet_predict -i ${ip} -o ${op} -t 607 -m 3d_fullres -f 4 -chk model_best -tr nnUNetTrainerV2_REUNet -p nnUNetPlans_REUNet_v2.1 --save_npz
```
