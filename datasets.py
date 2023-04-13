# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import json
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


def merge_batch_tensors_by_dict_key(batch):
    batch_tensors = {}
    for tensor_key in batch[0]:
        if isinstance(batch[0][tensor_key], torch.Tensor):
            batch_tensors[tensor_key] = torch.stack(
                [d[tensor_key] for d in batch])
        else:
            batch_tensors[tensor_key] = torch.tensor(
                [d[tensor_key] for d in batch], dtype=torch.long)
    return batch_tensors


class VQAv2Dataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, transform,
        tokenizer, num_max_bpe_tokens
    ):
        ans2label_file = os.path.join(data_path, "answer2label.txt")
        label2ans = []
        with open(ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                ans = data["answer"]
                label = data["label"]
                label = int(label)
                assert label == i
                label2ans.append(ans)

        self.label2ans = label2ans

        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        items = []

        data = {"image_path": "test2015/COCO_test2015_000000000001.jpg",
                "labels": [], "scores": [], "qid": 54668644678}
        question_text = 'What is the fence made of?'
        tokens = tokenizer.tokenize(question_text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        data['text_segment'] = token_ids
        items.append(data)

        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        t = type(image)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len):
        tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError(
                "The text segment should contains at least one tokens!")

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict, max_len):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(
            text_segment, max_len)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data, self.num_max_bpe_tokens)
        data["qid"] = self.items[index]["qid"]
        return data

    def __len__(self) -> int:
        return len(self.items)


def get_sentencepiece_model_for_beit3(sentencepiece_model_path):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(sentencepiece_model_path)


def get_dataset(input_size, data_path, num_max_bpe_tokens, sentencepiece_model_path):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN,
                             std=IMAGENET_INCEPTION_STD)
    ])
    tokenizer = get_sentencepiece_model_for_beit3(sentencepiece_model_path)
    dataset = VQAv2Dataset(
        data_path=data_path,
        transform=transform, tokenizer=tokenizer,
        num_max_bpe_tokens=num_max_bpe_tokens
    )
    return dataset


def create_dataset_by_split(args):
    input_size = args.input_size
    data_path = args.data_path
    num_max_bpe_tokens = args.num_max_bpe_tokens
    sentencepiece_model_path = args.sentencepiece_model

    dataset = get_dataset(input_size, data_path,
                          num_max_bpe_tokens, sentencepiece_model_path)

    ## Below is just to create batches
    sampler = torch.utils.data.SequentialSampler(dataset)

    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_mem = args.pin_mem
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        collate_fn=merge_batch_tensors_by_dict_key,
    )
    return dataloader
