# install REUnet by moving code
import nnunet
import os
import shutil
import sys
from setuptools import setup


def copytree(src, dst, symlinks=False, ignore=None):
    print('Copy {} to {}'.format(src, dst))
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def main():
    # set nnunet path
    nnu_dir = nnunet.__path__[0]
    preprocessed_dir = os.getenv('nnUNet_preprocessed')
    trained_model_path = os.getenv('RESULTS_FOLDER')

    REU_dir = sys.path[0]
    REU_model_dir = os.path.join(REU_dir, 'nnunet')
    REU_preprocessed_dir = os.path.join(REU_dir, 'nnUNet_preprocessed')
    REU_trained_model_dir = os.path.join(REU_dir, 'nnUnet_trained_models')

    # cp files
    copytree(REU_model_dir, nnu_dir)  # netrwork
    copytree(REU_preprocessed_dir, preprocessed_dir)  # preprocesssing

    print('nnu dir: {}'.format(nnu_dir))
    print('REU_model_dir: {}'.format(REU_model_dir))
    print('REU_preprocessed_dir: {}'.format(REU_preprocessed_dir))
    # print('REU_trained_model_dir: {}'.format(REU_trained_model_dir))
    print('REUnet is installed')
    
    # setup new cmd
    setup(
        entry_points = {
            "console_scripts": [
                "nnUNet_predict_sam = nnunet.inference.predict_sam_simple:main"
            ]
        }
    )


if __name__ == '__main__':
    main()

