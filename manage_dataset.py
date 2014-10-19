#!/bin/python
from __future__ import print_function, division
import argparse
import hashlib
import os
import os.path
import string
import sqlite3

# *_truth: true if it's confirmed by human
# state: {"empty", "up", "down"}
# type: {"empty", "FU", ..., "OU"}
#
# corners, config: JSON
#
# corners: 4 2D points as [[double]]
# config: {'xy': {state: state, type: type}
schema_photos = """
CREATE TABLE photos(
    id integer primary key,
    image blob,
    corners text, corners_truth bool,
    initial bool, initial_truth bool,
    config text, config_truth bool);
"""


def group_photos_by_content(conn):
    """
    Return {sha1 hexhash string: [row id]}
    """
    unique_rows = {}
    for (pid, image) in conn.execute('select id, image from photos'):
        m = hashlib.sha1()
        m.update(image)
        key = m.hexdigest()
        unique_rows.setdefault(key, []).append(pid)
    return unique_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
Merge new content into existing dataset, assigining safe unique keys.
File extensions will be lowercased.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'existing', metavar='DATASET_EXISTING', nargs=1, type=str,
        help='SQLite dataset path')
    # Merge new files into dataset.
    parser.add_argument(
        '--merge', metavar='DATASET_NEW', nargs='?', type=str, const=True,
        default=None,
        help='Merge new dataset directory into existing dataset (without duplication removal)')
    parser.set_defaults(is_initial=None)
    parser.add_argument('--is_initial', dest='is_initial', action='store_true')
    parser.add_argument('--is_not_initial', dest='is_initial', action='store_false')
    # Remove exact duplicates.
    parser.add_argument(
        '--find-duplicate', action='store_true',
        help='List duplicated ids. To run it, --remove-duplicate.')
    parser.add_argument(
        '--remove-duplicate', action='store_true',
        help='Remove duplicated ids')

    args = parser.parse_args()
    already_exist = os.path.isfile(args.existing[0])
    conn = sqlite3.connect(args.existing[0])
    if not already_exist:
        print('Creating table')
        conn.execute(schema_photos)
        conn.commit()

    if args.find_duplicate:
        unique_rows = group_photos_by_content(conn)
        duplicates = {
            k: ids for (k, ids) in unique_rows.items() if len(ids) > 1}
        print(duplicates)
        print("--remove-duplicate will select the smallest row id for each group")
    elif args.remove_duplicate:
        rows_to_remove = []
        for (k, ids) in group_photos_by_content(conn).items():
            rows_to_remove += sorted(ids)[1:]
        print("Removing these %d keys: %s" % (len(rows_to_remove), rows_to_remove))
        conn.executemany(
            "delete from photos where id = ?",
            [(rowid,) for rowid in rows_to_remove])
    elif args.merge is not None:
        for path in os.listdir(args.merge):
            path_src = os.path.join(args.merge, path)
            if not os.path.isfile(path_src):
                print('Ignoring %s (not file)' % path_src)
                continue

            print('Importing from %s with initial=%s' % (path_src, args.is_initial))
            conn.execute("""
                insert into photos (
                    image,
                    corners, corners_truth,
                    initial, initial_truth,
                    config, config_truth) values (
                    ?,
                    ?, ?,
                    ?, ?,
                    ?, ?) """, (
                buffer(open(path_src).read()),
                u"[]", False,
                False if args.is_initial is None else args.is_initial,
                False if args.is_initial is None else True,
                u"{}", False))
    conn.commit()

    print(
        'Number of photos now: %d' %
        conn.execute('select count(*) from photos').fetchone())
