import os
import re
import json
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import get_logger

logger = get_logger(__name__)

PAD_IDX = 42


class Tokenizer:
    def __init__(
        self,
        data_file: str | None = None,
        sp_model_prefix: str = None,
        vocab_size: int = 2000,
        normalization_rule_name: str = 'nmt_nfkc_cf',
        model_type: str = 'bpe',
        tokenizer_path: str | None = None,
    ):
        if data_file is not None and not os.path.isfile(sp_model_prefix + '.model'):
            SentencePieceTrainer.train(
                input=data_file,
                vocab_size=vocab_size,
                model_type=model_type,
                model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name,
                pad_id=PAD_IDX,
            )
        if tokenizer_path is not None:
            model_file = tokenizer_path
        else:
            model_file = sp_model_prefix + '.model'
        self.sp_model = SentencePieceProcessor(model_file=model_file)
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = (
            self.sp_model.pad_id(),
            self.sp_model.unk_id(),
            self.sp_model.bos_id(),
            self.sp_model.eos_id(),
        )

    def encode(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        return self.sp_model.encode(texts)

    def decode(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()
        return self.sp_model.decode(ids)


class TextDataset(Dataset):
    def __init__(self, data_file: str, tokenizer: Tokenizer, max_length: int = 1024, num_texts: None | int = None):
        self.tokenizer = tokenizer
        with open(data_file) as file:
            self.texts = file.readlines()
        if num_texts:
            self.texts = self.texts[:num_texts]
        self.indices = self.tokenizer.encode(self.texts)
        self.max_length = max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        indices = []
        indices.append(self.tokenizer.bos_id)
        indices.extend(self.indices[item][: self.max_length - 2])
        indices.append(self.tokenizer.eos_id)

        return torch.LongTensor(indices)


def collate_fn(batch):
    tokens = pad_sequence(
        batch,
        batch_first=True,
        padding_value=PAD_IDX,
    )
    targets = tokens[:, 1:].clone().detach()
    tokens = tokens[:, :-1].clone().detach()
    return tokens, targets


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src):
    src_seq_len = src.shape[1]
    mask = generate_square_subsequent_mask(src_seq_len)
    padding_mask = src == PAD_IDX
    return mask, padding_mask


def parse_jsons(config, all_data_filepath):
    _, _, json_names = next(os.walk(config.data_folder))
    with open(all_data_filepath, 'w') as out:
        for json_name in json_names:
            if not json_name.endswith(".json"):
                continue
            with open(config.data_folder + '/' + json_name, 'r') as f:
                json_data = json.load(f)
            for story in json_data:
                story = story["story"]
                story = re.sub('[^A-Za-z0-9.,!? \'\"]+', ' ', story)
                print(story, file=out, flush=True)


def get_dataloaders(config):
    all_data_filepath = f"{config.data_folder}/all_data.txt"

    if not os.path.isfile(all_data_filepath):
        logger.info(f"| {all_data_filepath} not found, parsing json files with texts")
        parse_jsons(config, all_data_filepath)
    else:
        logger.info(f"| using texts from {all_data_filepath}")

    tokenizer = Tokenizer(data_file=all_data_filepath, sp_model_prefix="bpe", vocab_size=config.vocab_size)
    dataset = TextDataset(all_data_filepath, tokenizer, max_length=config.max_len)

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.995, 0.005], generator=generator)

    train_dataloader = DataLoader(
        train_set, batch_size=config.bsz, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_set, batch_size=config.bsz, shuffle=False, num_workers=4, drop_last=True, collate_fn=collate_fn
    )

    return tokenizer, train_dataloader, val_dataloader
