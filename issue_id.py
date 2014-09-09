#!/bin/python
from __future__ import print_function, division
import argparse
import os
import os.path
import random
import string
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
Merge new content into existing dataset, assigining safe unique keys.
File extensions will be lowercased.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'new', metavar='DATASET_NEW', nargs=1, type=str,
        help='New dataset directory path')
    parser.add_argument(
        'existing', metavar='DATASET_EXISTING', nargs=1, type=str,
        help='Existing dataset directory path')

    args = parser.parse_args()
    if args.new == args.existing:
        raise SyntaxError('New and existing dataset cannot be the same')

    existing_ids = set(os.path.splitext(os.path.basename(path))[0] for path in os.listdir(args.existing[0]))

    def random_id(n):
        charset = string.lowercase + string.digits
        return ''.join(random.choice(charset) for i in range(n))

    def issue_new_id():
        n = 1
        while True:
            for i in range(3):
                i = random_id(n)
                if i not in existing_ids:
                    existing_ids.add(i)
                    return i
            else:
                n += 1

    for path in os.listdir(args.new[0]):
        entry_id = issue_new_id()
        path_src = os.path.join(args.new[0], path)
        path_dst = os.path.join(args.existing[0], entry_id + os.path.splitext(path)[1].lower())
        print('Copying from %s to %s' % (path_src, path_dst))
        shutil.copyfile(path_src, path_dst)
