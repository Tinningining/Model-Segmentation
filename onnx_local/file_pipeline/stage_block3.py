import argparse

from stage_block_common import build_parser, run_block


def main():
    parser = build_parser(block_idx=3)
    args = parser.parse_args()
    run_block(3, args)


if __name__ == "__main__":
    main()
