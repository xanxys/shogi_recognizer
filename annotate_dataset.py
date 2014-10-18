#!/bin/python2
from __future__ import print_function, division
import argparse
import base64
import json
import flask
import sqlite3

app = flask.Flask(__name__, template_folder='annotator/template')


@app.before_request
def before_request():
    print("Connecting to %s" % app.db_conn)
    flask.g.db = sqlite3.connect(app.db_conn)


@app.route("/")
def index():
    counts = {}
    counts["all"] = flask.g.db.execute('select count(*) from photos').fetchone()[0]
    counts["definitely_initial"] = flask.g.db.execute('select count(*) from photos where initial and initial_truth').fetchone()[0]

    result = []
    for (key, n) in counts.items():
        result.append('* %s: %d photos<br/>' % (key, n))
    stats_html = flask.Markup(''.join(result))

    result = []
    metadata = {}
    entries_per_page = 10
    for (i, (pid, image, corners, corners_truth, initial, initial_truth)) in enumerate(
            flask.g.db.execute('select id, image, corners, corners_truth, initial, initial_truth from photos')):
        if i >= entries_per_page:
            result += "...and more"
            break
        metadata[i] = {
            "corners": json.loads(corners),
            "corners_truth": corners_truth,
            "initial": initial,
            "initial_truth": initial_truth
        }
        image_data = "data:image/jpeg;base64,%s" % base64.b64encode(image)
        result.append("<div>")
        result.append("id=%d<br/>" % pid)
        result.append('<img src="%s"/>' % image_data)
        result.append("(%d byte)" % len(image))
        result.append(corners)
        result.append("</div>")
    photos_html = flask.Markup(''.join(result))

    return flask.render_template(
        "index.html",
        stats_html=stats_html,
        photos_html=photos_html,
        metadata=flask.Markup(json.dumps(metadata)))


@app.route('/static/<path>')
def serve_static(path):
    return flask.send_from_directory('./annotator/static', path)


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
