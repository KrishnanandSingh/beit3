# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import json
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


class BeitVQADataProcessor():
    def __init__(
        self, transform,
        tokenizer, num_max_bpe_tokens
    ):
        self.transform = transform
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens

        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def _get_image(self, image):
        return self.transform(image)

    def _get_text_segment(self, text_segment):
        tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError(
                "The text segment should contains at least one tokens!")

        if len(tokens) > self.num_max_bpe_tokens - 2:
            tokens = tokens[:self.num_max_bpe_tokens - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * \
            (self.num_max_bpe_tokens - num_tokens)
        return tokens + [self.pad_token_id] * (self.num_max_bpe_tokens - num_tokens), padding_mask, num_tokens

    def process_data(self, image, question: str, question_id: int):
        tokens = self.tokenizer.tokenize(question)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        data = dict()
        data["image"] = self._get_image(image)
        language_tokens, padding_mask, _ = self._get_text_segment(token_ids)
        data["language_tokens"] = torch.tensor(language_tokens)
        data["padding_mask"] = torch.tensor(padding_mask)
        data["qid"] = torch.tensor(question_id)
        return data


def get_sentencepiece_model_for_beit3(sentencepiece_model_path):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(sentencepiece_model_path)


def get_beit3_processor(input_size, ans2label_file_path, sentencepiece_model_path, num_max_bpe_tokens=64):
    label2ans = []
    with open(ans2label_file_path, mode="r", encoding="utf-8") as reader:
        for i, line in enumerate(reader):
            data = json.loads(line)
            ans = data["answer"]
            label = data["label"]
            label = int(label)
            assert label == i
            label2ans.append(ans)

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN,
                             std=IMAGENET_INCEPTION_STD)
    ])
    tokenizer = get_sentencepiece_model_for_beit3(sentencepiece_model_path)
    processor = BeitVQADataProcessor(
        transform=transform, tokenizer=tokenizer, num_max_bpe_tokens=num_max_bpe_tokens)
    return processor, label2ans
