#!/usr/bin/env python3
from espnet2.tasks.er import ERTask


def get_parser():
    parser = ERTask.get_parser()
    return parser


def main(cmd=None):
    r"""ER training.

    Example:

        % python er_train.py er --print_config --optim adadelta \
                > conf/train_er.yaml
        % python er_train.py --config conf/train_er.yaml
    """
    ERTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
