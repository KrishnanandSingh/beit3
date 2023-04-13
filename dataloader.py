'''
# Needs below files for generating index
# data/vqa-data/vqa/v2_OpenEnded_mscoco_train2014_questions.json
# data/vqa-data/vqa/v2_OpenEnded_mscoco_val2014_questions.json
# data/vqa-data/vqa/v2_mscoco_train2014_annotations.json
# data/vqa-data/vqa/v2_mscoco_val2014_annotations.json
# data/vqa-data/vqa/v2_OpenEnded_mscoco_test2015_questions.json
# data/vqa-data/vqa/v2_OpenEnded_mscoco_test-dev2015_questions.json (Not sure about this, it's a copy of test2015 questions)

# from datasets import VQAv2Dataset
# from transformers import XLMRobertaTokenizer

# tokenizer = XLMRobertaTokenizer("data/beit3.spm")

# VQAv2Dataset.make_dataset_index(
#     data_path="data/vqa-data",
#     tokenizer=tokenizer,
#     annotation_data_path="data/vqa-data/vqa",
# )

# setup
!git clone https://github.com/KrishnanandSingh/beit3
!cd beit3 && pip install -r requirements.txt
!wget https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm
# Downloading test data set with 2 images
!wget https://github.com/KrishnanandSingh/beit3/releases/download/v0.1.0-beta/vqa-data.tgz
!tar -xf vqa-data.tgz

'''
import datasets


def prepare_args():
    import argparse

    parser = argparse.ArgumentParser(
        'BEiT fine-tuning and evaluation script for image classification', add_help=False)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--sentencepiece_model', type=str, required=True,
                        help='Sentencepiece model path for the pretrained model.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    arg_list = [
        '--input_size', '480',
        '--batch_size', '16',
        '--sentencepiece_model', 'data/beit3.spm',
        '--data_path', 'data/vqa-data',
        '--num_workers', '1'
    ]
    args = parser.parse_args(arg_list)
    return args


def get_data_loader():
    args = prepare_args()
    data_loader = datasets.create_dataset_by_split(args)
    return data_loader


def get_dataset():
    args = prepare_args()
    input_size = args.input_size
    data_path = args.data_path
    num_max_bpe_tokens = args.num_max_bpe_tokens
    sentencepiece_model_path = args.sentencepiece_model
    dataset = datasets.get_dataset(
        input_size, data_path, num_max_bpe_tokens, sentencepiece_model_path)
    return dataset


if __name__ == '__main__':
    import torch
    dataset = get_dataset()
    data_item = dataset[0]
    data_item['qid'] = torch.tensor(data_item['qid'])
    data_item['language_tokens'] = torch.tensor(data_item['language_tokens'])
    data_item['padding_mask'] = torch.tensor(data_item['padding_mask'])
    for tensor_key in data_item.keys():
        data_item[tensor_key] = torch.stack([data_item[tensor_key]])

    data_loader = get_data_loader()
    for data in data_loader:
        for tensor_key in data.keys():
            data[tensor_key] = torch.stack([data[tensor_key][0]])
    print(data)
    print(data_item)