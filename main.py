import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Incident prediction from time-series metrics")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Config: W={args.window_size}, H={args.horizon}, seed={args.seed}")

    # TODO: wire up pipeline


if __name__ == "__main__":
    main()
