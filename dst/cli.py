import argparse
import json
from .tokenizer import DSTTokenizer


def main():
    parser = argparse.ArgumentParser(description="Domain-Specific Tokenization CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--input", required=True, help="Path to training corpus")
    train_p.add_argument("--output", default="dst_tokenizer.json")
    train_p.add_argument("--min-freq", type=int, default=5, help="Minimum frequency for candidate tokens")
    train_p.add_argument("--max-vocab", type=int, default=32000, help="Maximum vocabulary size")

    encode_p = sub.add_parser("encode")
    encode_p.add_argument("--input", required=True)
    encode_p.add_argument("--tokenizer", default="dst_tokenizer.json")
    encode_p.add_argument("--format", choices=["plain", "json"], default="plain", help="Output format for tokens")

    args = parser.parse_args()

    if args.cmd == "train":
        with open(args.input, "r", encoding="utf-8") as f:
            corpus = f.read().splitlines()
        tokenizer = DSTTokenizer.train(
            corpus,
            min_freq=args.min_freq,
            max_vocab=args.max_vocab,
        )
        tokenizer.save_json(args.output)
        print(f"✅ Tokenizer trained and saved to {args.output}")

    elif args.cmd == "encode":
        tokenizer = DSTTokenizer.load_json(args.tokenizer)
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                tokens = tokenizer.encode(line)
                if args.format == "json":
                    print(json.dumps(tokens, ensure_ascii=False))
                else:
                    print(" ".join(tokens))


if __name__ == "__main__":
    main()
