#!/bin/python2
from __future__ import print_function, division
import argparse
import base64
import json
import flask
import sqlite3
import re
import shogi

app = flask.Flask(__name__, template_folder='annotator/template')


class APIUsageError(Exception):
    def __init__(self, message):
        Exception.__init__(self)
        self.message = message


@app.before_request
def before_request():
    print("Connecting to %s" % app.db_conn)
    flask.g.db = sqlite3.connect(app.db_conn)


@app.route("/")
def index():
    counts = {}
    counts["all"] = flask.g.db.execute('select count(*) from photos').fetchone()[0]
    counts["definitely_initial"] = flask.g.db.execute('select count(*) from photos where initial and initial_truth').fetchone()[0]
    counts["corners_is_truth"] = flask.g.db.execute('select count(*) from photos where corners_truth').fetchone()[0]
    counts["config_is_truth"] = flask.g.db.execute('select count(*) from photos where config_truth').fetchone()[0]

    result = []
    for (key, n) in counts.items():
        result.append('* %s: %d photos<br/>' % (key, n))
    stats_html = flask.Markup(''.join(result))

    result = []
    entries_per_page = 10
    for (i, (pid, image, corners, corners_truth, initial, initial_truth)) in enumerate(
            flask.g.db.execute('select id, image, corners, corners_truth, initial, initial_truth from photos')):
        if i >= entries_per_page:
            result += "...and more"
            break
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
        photos_html=photos_html)


@app.route("/photos")
def photos():
    """
    Don't return images themselves.
    """
    filter_unc_corners = flask.request.args.get('uncertain_corners') is not None
    filter_unc_config = flask.request.args.get('uncertain_config') is not None

    results = []
    for (pid, corners, corners_truth, initial, initial_truth, config, config_truth) in flask.g.db.execute(
            'select id, corners, corners_truth, initial, initial_truth, config, config_truth from photos'):
        if filter_unc_corners and corners_truth:
            continue
        if filter_unc_config and config_truth:
            continue
        results.append({
            "id": pid,
            "corners": json.loads(corners),
            "corners_truth": corners_truth,
            "initial": initial,
            "initial_truth": initial_truth
        })
    return flask.jsonify(results=results)


@app.route("/photo/<int:photo_id>")
def photo(photo_id):
    entry = flask.g.db.execute(
        """select
            image,
            corners, corners_truth,
            initial, initial_truth,
            config, config_truth
            from photos where id = ?""",
        (photo_id,)).fetchone()
    if entry is None:
        flask.abort(404)
    (image,
        corners, corners_truth,
        initial, initial_truth,
        config, config_truth) = entry
    image_data = "data:image/jpeg;base64,%s" % base64.b64encode(image)
    data = {
        "id": photo_id,
        "image_uri_encoded": image_data,
        "corners": json.loads(corners),
        "corners_truth": corners_truth,
        "initial": initial,
        "initial_truth": initial_truth,
        "config": json.loads(config),
        "config_truth": config_truth
    }
    return flask.jsonify(**data)


@app.route("/photo/<int:photo_id>", methods=["POST"])
def post_photo(photo_id):
    raw_query = flask.request.get_json()
    if 'corners' in raw_query:
        raw_corners = raw_query['corners']
        # sanitize
        if len(raw_corners) != 4:
            flask.abort(403)
        if any(len(corner) != 2 for corner in raw_corners):
            flask.abort(403)
        corners = raw_corners
        # store
        flask.g.db.execute(
            'update photos set corners_truth = 1, corners=? where id = ?',
            (json.dumps(corners), photo_id))
    if 'config' in raw_query:
        raw_config = raw_query['config']
        if type(raw_config) is not dict:
            raise APIUsageError('config must be object')
        all_types = shogi.all_types | set(['empty'])
        for (k, v) in raw_config.items():
            if re.match('^[1-9][1-9]$', k) is None:
                raise APIUsageError('Invalid key: %s' % k)
            if v['state'] not in ['empty', 'down', 'up']:
                raise APIUsageError('Invalid state: %s' % v['state'])
            if v['type'] not in all_types:
                raise APIUsageError('Invalid type: %s' % v['type'])
            if v['state'] == 'empty' and v['type'] != 'empty':
                raise APIUsageError(
                    'Contradictory state(%s) and type(%s)' %
                    (v['state'], v['empty']))
            if v['state'] != 'empty' and v['type'] == 'empty':
                raise APIUsageError(
                    'Contradictory state(%s) and type(%s)' %
                    (v['state'], v['empty']))
        config = raw_config
        # store
        flask.g.db.execute(
            'update photos set config_truth = 1, config=? where id = ?',
            (json.dumps(config), photo_id))
    flask.g.db.commit()
    return flask.jsonify(status="success")


@app.route('/static/<path>')
def serve_static(path):
    return flask.send_from_directory('./annotator/static', path)


@app.errorhandler(APIUsageError)
def error_api_usage(exc):
    response = flask.jsonify({
        "status": "error",
        "message": exc.message
    })
    response.status_code = 403
    return response

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
