#!/bin/python2
from __future__ import print_function, division
import argparse
import flask
import sqlite3

app = flask.Flask(__name__)


@app.before_request
def before_request():
    print("Connecting to %s" % app.db_conn)
    flask.g.db = sqlite3.connect(app.db_conn)


@app.route("/")
def hello():
    counts = {}
    counts["all"] = flask.g.db.execute('select count(*) from photos').fetchone()[0]
    counts["definitely_initial"] = flask.g.db.execute('select count(*) from photos where initial and initial_truth').fetchone()[0]

    result = "photos<br/>"
    for (key, n) in counts.items():
        result += '* %s: %d photos<br/>' % (key, n)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Launch a web interface to view / annotate datasets.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset', metavar='DATASET_EXISTING', nargs=1, type=str,
        help='SQLite dataset path')

    args = parser.parse_args()
    app.db_conn = args.dataset[0]
    app.run(port=8374)
