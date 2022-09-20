import requests
import os
import sys
from tqdm import tqdm


def download_zip_file(url, zip_download_path):
    with requests.get(url, stream=True, timeout=200) as r:
        r.raise_for_status()
        with tqdm.wrapattr(open(zip_download_path, "wb"), "write", miniters=1,
                           total=int(r.headers.get('content-length', 0)),
                           desc=zip_download_path) as fout:
            for chunk in r.iter_content(chunk_size=4096):
                fout.write(chunk)

    return zip_download_path


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
    zip_download_path = download_zip_file(url, zip_download_path)
    print('Temporary zip file downloaded， starting to Unzip...')

    # unzip
    assert os.path.isfile(zip_download_path), '{} Not a file!'.format(zip_download_path)

    os.system('unzip {} -d {}'.format(zip_download_path, trained_model_path))
    print('Unzip is complete, starting to removing temporary file...')

    os.remove(zip_download_path)
    print('Removing complete')
    print('REUnet trained models downloaded successfully!')


if __name__ == '__main__':
    main()
