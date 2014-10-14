#!/bin/python2
from __future__ import print_function, division
import argparse
import base64
import flask
import sqlite3

app = flask.Flask(__name__)


@app.before_request
def before_request():
    print("Connecting to %s" % app.db_conn)
    flask.g.db = sqlite3.connect(app.db_conn)


@app.route("/")
def index():
    counts = {}
    counts["all"] = flask.g.db.execute('select count(*) from photos').fetchone()[0]
    counts["definitely_initial"] = flask.g.db.execute('select count(*) from photos where initial and initial_truth').fetchone()[0]

    result = ""
    result += "<h1>Stats</h1>"
    for (key, n) in counts.items():
        result += '* %s: %d photos<br/>' % (key, n)

    result += "<h1>Photos</h1>"
    entries_per_page = 100
    for (i, row) in enumerate(flask.g.db.execute('select id, image from photos')):
        if i >= entries_per_page:
            result += "...and more"
            break
        pid, image = row

        image_data = "data:image/jpeg;base64,%s" % base64.b64encode(image)
        result += "<div>"
        result += "id=%d<br/>" % pid
        result += '<img src="%s"/>' % image_data
        result += "(%d byte)" % len(image)
        result += "</div>"

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
    app.run(port=8374, debug=True)
