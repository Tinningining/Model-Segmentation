import argparse

from stage_block_common import build_parser, run_block


def main():
    parser = build_parser(block_idx=2)
    args = parser.parse_args()
    run_block(2, args)


if __name__ == "__main__":
    main()
