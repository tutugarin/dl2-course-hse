import argparse
import sys
import torch

from time import perf_counter

from dataset import Tokenizer
from model import LanguageModel
from utils import Config, get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='GPT tiny stories inference')
    parser.add_argument('--promt', type=str, default='', help='beginning of the story')

    parser.add_argument('--checkpoint-path', type=str, default='checkpoint_best.pt', help='model checkpoint')
    parser.add_argument('--tokenizer-path', type=str, default='bpe.model', help='bpe model')

    parser.add_argument('--output-file', default=None, help='path to file where to write generated samples')
    parser.add_argument('--gen-temperature', type=float, default=1, help='generation temperature')
    parser.add_argument('--gen-topk', type=int, default=1, help='number of tokens to select from')
    parser.add_argument('--gen-quantity', type=int, default=1, help='quantity samples to generate')

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()
    model = LanguageModel(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    state = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(state["model"])
    model = model.to(config.device)
    tokenizer = Tokenizer(tokenizer_path=args.tokenizer_path)

    output_file = open(args.output_file, 'w') if args.output_file is not None else sys.stdout
    for i in range(1, args.gen_quantity + 1):
        start_generation = perf_counter()
        logger.info(f"| generating sample #{i} started")
        print(
            model.inference(
                tokenizer,
                args.promt,
                max_new_tokens=config.max_len,
                temperature=args.gen_temperature,
                top_k=args.gen_topk,
            ),
            file=output_file,
        )
        logger.info(f"| generating sample #{i} total time: {perf_counter() - start_generation}")
    if args.output_file is not None:
        output_file.close()


if __name__ == "__main__":
    main()
