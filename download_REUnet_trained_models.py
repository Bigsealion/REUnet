import requests
import os
import sys
from tqdm import tqdm


def main():
    # set parameter
    url = 'https://zenodo.org/record/7091314/files/REUNet_trained_models.zip?download=1'
    zip_download_path = os.path.join(sys.path[0], 'REUnet_trained_models.zip')

    # nnUnet trained model path
    trained_model_path = os.getenv('RESULTS_FOLDER')
    assert trained_model_path is not None, 'nnUnet RESULTS_FOLDER Not Found!'

    # download
    print('Temporary files will be downloaded to\n\t{}'.format(zip_download_path))
    print('And will be removed afterd unzip.')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tqdm.wrapattr(open(zip_download_path, "wb"), "write", miniters=1,
                           total=int(r.headers.get('content-length', 0)),
                           desc=zip_download_path) as fout:
            for chunk in r.iter_content(chunk_size=4096):
                fout.write(chunk)

    # unzip
    os.system('unzip {} -d {}'.format(zip_download_path, trained_model_path))
    os.remove(zip_download_path)


if __name__ == '__main__':
    main()
