# REUnet
REUnet, an externally validated deep learning model for fine segmentation of intracerebral and intraventricular hemorrhage on head CT images 

# Requirements
Install [`nnU-Net`](https://github.com/MIC-DKFZ/nnUNet), which is the framework for running REUnet.

Download [`StripSkullCT`](https://github.com/WuChanada/StripSkullCT) for brain extraction.

# Installation
TODO

# Usage
After installation, you can get ICH and IVH segmentation by following steps:

1. Saving your brain CT image by `Nifti` format

2. Applying brain extraction by [`StripSkullCT`](https://github.com/WuChanada/StripSkullCT) in `Matlab`. 

3. Running the following command



