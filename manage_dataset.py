#!/bin/python
from __future__ import print_function, division
import argparse
import os
import os.path
import string
import sqlite3

# *_truth: true if it's confirmed by human
# state: {"empty", "up", "down"}
# type: {"empty", "FU", ..., "OU"}
schema_photos = """
CREATE TABLE photos(
    id integer primary key,
    image blob,
    corners text, corners_truth bool,
    initial bool, initial_truth bool);
"""

schema_cells = """
CREATE TABLE cells(
    id integer primary key,
    photo_id integer, i integer, j integer,
    state text, state_truth bool,
    type text, type_truth bool);
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
Merge new content into existing dataset, assigining safe unique keys.
File extensions will be lowercased.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'existing', metavar='DATASET_EXISTING', nargs=1, type=str,
        help='SQLite dataset path')
    parser.add_argument(
        'new', metavar='DATASET_NEW', nargs=1, type=str,
        help='New dataset directory path')
    parser.set_defaults(is_initial=None)
    parser.add_argument('--is_initial', dest='is_initial', action='store_true')
    parser.add_argument('--is_not_initial', dest='is_initial', action='store_false')

    args = parser.parse_args()
    if args.new == args.existing:
        raise SyntaxError('New and existing dataset cannot be the same')

    already_exist = os.path.isfile(args.existing[0])
    conn = sqlite3.connect(args.existing[0])
    if not already_exist:
        print('Creating tables')
        conn.execute(schema_photos)
        conn.execute(schema_cells)
        conn.commit()

    for path in os.listdir(args.new[0]):
        path_src = os.path.join(args.new[0], path)
        if not os.path.isfile(path_src):
            print('Ignoring %s (not file)' % path_src)
            continue

        print('Importing from %s with initial=%s' % (path_src, args.is_initial))
        conn.execute("""
            insert into photos (
                image,
                corners, corners_truth,
                initial, initial_truth) values (
                ?,
                ?, ?,
                ?, ?) """, (
            buffer(open(path_src).read()),
            u"[]", False,
            False if args.is_initial is None else args.is_initial,
            False if args.is_initial is None else True))
    conn.commit()
    print(
        'Number of photos now: %d' %
        conn.execute('select count(*) from photos').fetchone())
