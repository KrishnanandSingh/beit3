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


if __name__ == '__main__':
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
    question_text = question

    data_item = processor.process_data(image, question, question_id)
    import torch
    for tensor_key in data_item.keys():
        data_item[tensor_key] = torch.stack([data_item[tensor_key]])

    print(data_item)