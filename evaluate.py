'''
# Use python3
python -m venv beit-pi
source ~/beit-pi/bin/activate
git clone --branch data-processor --depth 1 https://github.com/KrishnanandSingh/beit3.git
cd beit3
pip install -r requirements.txt
python evaluate.py

# If running on raspberry pi Ubuntu, do
pip install -r requirements-pi.txt

'''
import hashlib
import os
import shutil
import sys
import zipfile
from contextlib import contextmanager
from timeit import default_timer

import requests
import torch

import datasets


@contextmanager
def elapsed_timer():
    start = default_timer()
    def elapser(): return default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    def elapser(): return end-start


def __download_hook(count, block_size, total_size):
    """A hook to report the download progress."""
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write(f'\rDownloading: {percent}%')
    sys.stdout.flush()


def download(url, dir, unzip=False, extracted_file_name=None):
    filename = url.split('/')[-1]
    file_path = os.path.join(dir, filename)
    os.makedirs(dir, exist_ok=True)

    print(f'Downloading: {url}\n')
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024*1024
    with open(file_path, "wb") as f:
        chunk_count = 0
        for chunk in response.iter_content(chunk_size=block_size):
            # Update progress bar
            chunk_count += 1
            __download_hook(chunk_count, block_size, total_size)
            # Write data to file
            f.write(chunk)

    if unzip is True:
        print(f'\nExtracting : {file_path}')
        shutil.unpack_archive(file_path, dir, format='zip')
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # If there is only one file in zip and user wants to rename it
            if len(zip_ref.namelist()) == 1 and extracted_file_name:
                print('renaming file')
                extracted_file_path = os.path.join(dir, extracted_file_name)
                shutil.move(src=os.path.join(
                    dir, zip_ref.namelist()[0]), dst=extracted_file_path)


def calculate_md5_checksum(file_path, chunk_size=8192):
    hash_object = hashlib.md5()
    # Read the file in chunks and update the hash object incrementally
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            hash_object.update(data)
    checksum = hash_object.hexdigest()
    return checksum


def download_sentence_piece_model(download_dir):
    sentence_piece_model_url = 'https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm'
    file_name = sentence_piece_model_url.split('/')[-1]
    if not os.path.isfile(os.path.join(download_dir, file_name)):
        download(sentence_piece_model_url, download_dir)


def download_ans2label(download_dir):
    ans2label_url = 'https://github.com/KrishnanandSingh/beit3/releases/download/v0.2.0/answer2label.zip'
    if not os.path.isfile(os.path.join(download_dir, 'answer2label.txt')):
        download(ans2label_url, download_dir, unzip=True,
                 extracted_file_name='answer2label.txt')


def download_large_model(download_dir):
    beit_large_model_md5_url = 'https://github.com/KrishnanandSingh/beit3/releases/download/v0.2.0/model.md5'
    md5_file_name = beit_large_model_md5_url.split('/')[-1]
    md5_file_path = os.path.join(download_dir, md5_file_name)
    if not os.path.isfile(md5_file_path):
        download(beit_large_model_md5_url, download_dir)
    required_md5_checksum = open(md5_file_path).read()

    beit_large_model_url = 'https://github.com/KrishnanandSingh/beit3/releases/download/v0.2.0/model.pth'
    model_file_name = beit_large_model_url.split('/')[-1]
    model_file_path = os.path.join(download_dir, model_file_name)
    if not os.path.isfile(model_file_path):
        download(beit_large_model_url, download_dir)

    print('Ensuring checksum of the large model')
    checksum = calculate_md5_checksum(model_file_path)
    if checksum != required_md5_checksum:
        print(
            f'model file corrupt: {checksum} != {required_md5_checksum} Redownloading..')
        os.remove(model_file_path)
        download(beit_large_model_url, download_dir)


def ensure_pre_requisites():
    download_dir = 'data'
    os.makedirs(download_dir, exist_ok=True)
    download_sentence_piece_model(download_dir)
    download_ans2label(download_dir)
    download_large_model(download_dir)
    print('Pre-requisites files check complete')


def load_model(device):
    print('Loading model')
    beit_model = torch.load('data/model.pth', map_location=device)
    model = beit_model
    model.to(device)
    model.eval()
    print(f'Model loaded on device {device}, ready to infer')
    return model


def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f'using {device}')
    return device


def one_dataset():
    ensure_pre_requisites()
    device = setup_device()
    model = load_model(device)
    input_size = 480
    ans2label_file_path = 'data/answer2label.txt'
    sentencepiece_model_path = 'data/beit3.spm'
    processor, label2ans = datasets.get_beit3_processor(
        input_size, ans2label_file_path, sentencepiece_model_path)

    import requests
    from PIL import Image

    print('Downloading image')
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    question = 'What is the cat doing?'
    question_id = 54668644678
    print('Preprocessing')
    data_item = processor.process_data(image, question, question_id)
    print('Inferring')
    prediction = None
    with elapsed_timer() as elapsed:
        with torch.no_grad():
            for tensor_key in data_item.keys():
                data_item[tensor_key] = torch.stack([data_item[tensor_key]])
            logits = model(
                image=data_item['image'], question=data_item['language_tokens'], padding_mask=data_item['padding_mask'])
            _, preds = logits.max(-1)
            prediction = {
                "question_id": data_item['qid'][0].item(),
                "answer":  label2ans[preds[0].item()],
            }
    print(
        f'On device {device.type}, Took {elapsed():.3f}s, Prediction: {prediction}')


if __name__ == '__main__':
    print('Using single item from dataset')
    one_dataset()
