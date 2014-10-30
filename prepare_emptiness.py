#!/bin/python2
from __future__ import print_function, division
import argparse
import os
import os.path
import hashlib
import random


def dump_sample_list(fobj, ls):
    """
    ls: [(image file path, label id)]
    """
    for entry in ls:
        fobj.write('%s %d\n' % entry)


if __name__ == '__main__':
    # for reproducibility
    random.seed(1)

    in_dir_path = 'derived/cells-emptiness'
    out_train_list_path = 'temp-input-train.txt'
    out_test_list_path = 'temp-input-test.txt'
    train_ratio = 0.8

    labels = ["occupied", "empty"]
    label_to_id = {l: i for (i, l) in enumerate(labels)}

    # Create list
    ls = []
    hashes = set()
    for p in os.listdir(in_dir_path):
        full_path = os.path.join(in_dir_path, p)
        img_hash = hashlib.sha1(open(full_path, 'rb').read()).hexdigest()
        if img_hash in hashes:
            print('Ignoring duplicate')
            continue
        hashes.add(img_hash)
        label = p.split('.')[-2].split('-')[-1]
        label_id = label_to_id[label]
        ls.append((full_path, label_id))

    # Split and write
    random.shuffle(ls)
    n_train = int(len(ls) * train_ratio)
    n_test = len(ls) - n_train
    dump_sample_list(open(out_train_list_path, 'w'), ls[:n_train])
    dump_sample_list(open(out_test_list_path, 'w'), ls[n_train:])
    print('%d training samples + %d test samples written' % (n_train, n_test))
