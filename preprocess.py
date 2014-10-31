#!/bin/python
from __future__ import print_function, division
import argparse
import numpy as np
import math
import scipy.linalg as la
import cv
import cv2
import os
import os.path
import random
import multiprocessing
import traceback
import itertools
import classify
import cairo
import sqlite3
import lsd
import json
import shogi


def clean_directory(dir_path):
    """
    Delete all files in dir_path.
    """
    for path in os.listdir(dir_path):
        os.unlink(os.path.join(dir_path, path))


def get_rot_invariants_initial():
    """
    Return positions invariant to 90-degree rotation,
    only considering empty vs occupied categories.

    (always_empty, always_occupied)
    """
    initial_state = shogi.get_initial_configuration()
    common_occupied = set()
    common_empty = set()
    for i in range(1, 10):
        for j in range(1, 10):
            p = (i, j)
            pt = (j, i)
            if p in initial_state and pt in initial_state:
                common_occupied.add(p)
            elif p not in initial_state and pt not in initial_state:
                common_empty.add(p)
    return (common_empty, common_occupied)


def rhotheta_to_cartesian(rho, theta):
    """
    Return (org, dir)
    """
    line_d = np.array([-math.sin(theta), math.cos(theta)])
    line_n = np.array([math.cos(theta), math.sin(theta)])
    return (line_n * rho, line_d)


def intersect_lines(l0, l1):
    """
    Return intersection point coordinate of two lines.
    """
    o0, d0 = rhotheta_to_cartesian(*l0)
    o1, d1 = rhotheta_to_cartesian(*l1)

    # solve (t0, t1) in o0 + d0 * t0 = o1 + d1 * t1
    # (d0, -d1) (t0, t1)^T = o1 - o0
    m = np.array([d0, -d1]).T
    v = o1 - o0
    try:
        ts = la.solve(m, v)
        return o0 + d0 * ts[0]
    except la.LinAlgError:
        # lines are parallel or some other degenerate case
        return d0 * 1e6


def point_to_direction(focal_px, center, pt):
    """
    Return 3d unit vector in camera coordinates.
    """
    x, y = pt - center
    d = np.array([x, y, focal_px])
    d /= la.norm(d)
    return d


def detect_board_vps(photo_id, img_shape, lines, visualize):
    """
    Detect 2 VPs of shogi board in img.
    VPs are represented as unit 3D vector in camera coordinates.

    return (hfov, (VP0, VP1))
    """
    assert(len(lines) >= 4)
    hfov_min = 0.2
    hfov_max = 1.5
    n_hfov_step = 5

    max_n_inliers = 0
    max_ns = None
    max_inliers = None
    max_fov = None
    for hfov in np.linspace(hfov_min, hfov_max, num=n_hfov_step):
        focal_px = img_shape[0] / 2 / math.tan(hfov / 2)
        center = np.array([img_shape[1], img_shape[0]]) / 2
        lines_normals = []
        for line in lines:
            l_org, l_dir = rhotheta_to_cartesian(*line)
            p0 = point_to_direction(focal_px, center, l_org - l_dir * 100)
            p1 = point_to_direction(focal_px, center, l_org + l_dir * 100)
            n = np.cross(p0, p1)
            n /= la.norm(n)
            lines_normals.append(n)

        # Use 3-RANSAC
        n_iter_vp = 3000
        dist_thresh = 0.01

        for i in range(n_iter_vp):
            line_tri = random.sample(lines_normals, 3)
            # First VP (a pole on sphere)
            n0 = np.cross(line_tri[0], line_tri[1])
            n0 /= la.norm(n0)
            # Second VP
            n1 = np.cross(n0, line_tri[2])
            # smaller sin is good.
            if la.norm(n1) > 0.9:
                continue  # ignore unreliable sample
            n1 /= la.norm(n1)
            # Calculate inliers.
            # (Since images will contain less height component,
            # we ignore 3rd VP.)
            n_inliers = 0
            for ln in lines_normals:
                dist0 = abs(math.asin(np.dot(ln, n0)))
                dist1 = abs(math.asin(np.dot(ln, n1)))
                if dist0 < dist_thresh and dist1 < dist_thresh:
                    continue
                if dist0 < dist_thresh or dist1 < dist_thresh:
                    n_inliers += 1
            if n_inliers > max_n_inliers:
                inl0 = []
                inl1 = []
                for (lorg, ln) in zip(lines, lines_normals):
                    dist0 = abs(math.asin(np.dot(ln, n0)))
                    dist1 = abs(math.asin(np.dot(ln, n1)))
                    if dist0 < dist_thresh and dist1 < dist_thresh:
                        continue
                    if dist0 < dist_thresh:
                        inl0.append(lorg)
                    if dist1 < dist_thresh:
                        inl1.append(lorg)
                # Final validation
                # avoid bundled line segments;
                # probably they're some singularity
                if np.std(np.array(inl0)[:, 0]) < 20:
                    continue
                if np.std(np.array(inl1)[:, 0]) < 20:
                    continue
                max_n_inliers = n_inliers
                max_ns = (n0, n1)
                max_inliers = (inl0, inl1)
                max_fov = hfov
    print("Max: fov=%.1f #inl=%d axis=%s" % (max_fov, max_n_inliers, max_ns))
    # if visualize:
    #     img_vps = np.copy(img)
    #     for (ix, color) in [(0, (0, 0, 255)), (1, (0, 255, 0))]:
    #         for line in max_inliers[ix]:
    #             draw_rhotheta_line(img_vps, line, color)
    #     cv2.imwrite('debug/proc-%s-vps.png' % photo_id, img_vps)
    return (max_fov, max_ns)


def find_9segments(xs, valid_width_range):
    """
    Find 10 values (with 9 intervals) in array of scalars.
    return: candidates [10 values in increasing order]
    """
    min_dx, max_dx = valid_width_range
    ratio_thresh = 0.08
    good_seps = []
    xs.sort()

    for (x0, x1) in itertools.combinations(xs, 2):
        dx = abs(x1 - x0) / 9
        if not (min_dx <= dx <= max_dx):
            continue
        segs = {}
        for x in xs:
            t = (x - x0) / dx
            key = int(round(t))
            dt = abs(t - key)
            if dt < ratio_thresh:
                segs.setdefault(key, []).append(x)

        # Find 10-continuous keys
        if len(segs) >= 10:
            # Split into continuous segments
            ks = sorted(segs.keys())
            cont = [ks[0]]
            cont_segs = []
            for (prev_k, k) in zip(ks, ks[1:]):
                if k == prev_k + 1:
                    cont.append(k)
                else:
                    cont_segs.append(cont)
                    cont = [k]
            if len(cont) >= 1:
                cont_segs.append(cont)

            # Discard smaller than 10 segs
            cont_segs = filter(lambda s: len(s) >= 10, cont_segs)

            # Now report all 10-segment in cont_segs as candidate
            for cs in cont_segs:
                vs = [np.mean(segs[k]) for k in cs]
                for i in range(len(cs) - 10 + 1):
                    good_seps.append(vs[i:i+10])
    # Although each seps contains 10 values,
    # effective dimension is just g2.
    # TODO: Bundle similar seps to reduce load in later steps
    return good_seps


def visualize_1d(arr, normalize=True, height=50):
    """
    Create horizontally long strip image to visualize 1D
    array.
    """
    width = len(arr)
    img = arr.reshape([1, width])
    if normalize:
        img *= 255 / max(arr)
    return cv2.resize(img, (width, height))


def detect_board_grid(photo_id, img, img_gray, region, visualize):
    """
    Detect shogi board pattern in lines.

    * img: size reference and background for visualization
    * region: 4 corners

    Use "3-line RANSAC" with variable hfov.
    (it operates on surface of a sphere)

    The center of img must be forward direction (i.e. img must not be cropped)

    return (orthogonal_image, grid_desc, perspective_trans)
    """
    assert(len(region) == 4)
    depersp_size = 900
    margin = 5
    pts_photo = region
    pts_correct = []
    for (ix, iy) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
        pts_correct.append(np.array([
            margin + ix * (depersp_size - 2 * margin),
            margin + iy * (depersp_size - 2 * margin)]))

    trans_persp = cv2.getPerspectiveTransform(
        np.array(pts_photo).astype(np.float32), np.array(pts_correct).astype(np.float32))
    # Correct perspectiveness.
    # This will result in orthogonal image with elongation.
    img_depersp = cv2.warpPerspective(img, trans_persp, (depersp_size, depersp_size), borderMode=cv2.BORDER_REPLICATE)
    if la.det(trans_persp) < 0:
        img_depersp = img_depersp[:, ::-1, :]
    if visualize:
        cv2.imwrite('debug/proc-%s-depersp.png' % photo_id, img_depersp)
    # Now we can treat X and Y axis separately.
    # Detect 10 lines in X direction
    min_dx = (depersp_size - margin * 2) / 2 / 9  # assume at least half of the image is covered by board
    max_dx = (depersp_size - margin * 2) / 9
    img_gray = cv2.cvtColor(img_depersp, cv.CV_BGR2GRAY)
    thresh = 5
    ls = map(lambda l: l[:4], lsd.detect_line_segments(img_gray.astype(np.float64), log_eps=-1))
    ls_x = []
    ls_y = []
    ls_others = []
    for l in ls:
        l = map(int, l)
        x0, y0, x1, y1 = l
        length = la.norm(np.array([x0 - x1, y0 - y1]))
        # reject too short segments (most likely letters and piece boundaries)
        if length < min_dx:
            continue
        # reject too long segments (LSD will detect both sides of grid lines,
        # so cell segments will be shorter than 1 cell)
        if length > max_dx:
            continue

        if abs(x1 - x0) < thresh:
            ls_x.append(l)
        elif abs(y1 - y0) < thresh:
            ls_y.append(l)
        else:
            ls_others.append(l)

    print('OrthoLine:%d X:%d Y:%d' % (len(ls), len(ls_x), len(ls_y)))
    if visualize:
        img_debug = cv2.cvtColor(img_gray, cv.CV_GRAY2BGR)
        for (x0, y0, x1, y1) in ls_x:
            cv2.line(img_debug, (x0, y0), (x1, y1), (0, 0, 255), thickness=2)
        for (x0, y0, x1, y1) in ls_y:
            cv2.line(img_debug, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
        for (x0, y0, x1, y1) in ls_others:
            cv2.line(img_debug, (x0, y0), (x1, y1), (255, 0, 0))
        cv2.imwrite('debug/proc-%s-ortho.png' % photo_id, img_debug)
    if len(ls_x) < 10 or len(ls_y) < 10:
        print('WARN: not enough XY lines')
        return None
    # Detect repetition in each axis
    xs = map(lambda line: line[0], ls_x)
    ys = map(lambda line: line[1], ls_y)
    if visualize:
        import matplotlib.pyplot as plt
        dxs = []
        for (x0, x1) in itertools.combinations(xs, 2):
            dxs.append(abs(x1 - x0))

        plt.figure(photo_id)
        plt.hist(dxs, bins=2000)
        plt.axvline(min_dx)
        plt.axvline(max_dx)
        plt.xlim(min_dx * 0.8, max_dx * 1.2)
        plt.savefig('debug/hist-dx-%s.png' % photo_id)

    def get_probable(vs):
        dvs = []
        for (v0, v1) in itertools.combinations(vs, 2):
            dvs.append(abs(v1 - v0))
        hist, bin_edges = np.histogram(dvs, bins=2000)
        return max(
            filter(lambda (h, be): min_dx < be < max_dx, zip(hist, bin_edges)),
            key=lambda (h, be): h)[1]

    def get_rep(vs):
        w_size = depersp_size
        assert(max(vs) < w_size)
        arr = np.zeros([w_size], np.float32)
        for v in vs:
            arr[int(v)] += 1
        cv2.imwrite('debug/imp-%s.png' % photo_id, visualize_1d(arr))
        freq = np.abs(np.fft.rfft(arr))
        freq[0] = 0  # delete DC component
        cv2.imwrite('debug/fourier-%s.png' % photo_id, visualize_1d(freq))

        return np.argmax(freq) * 2

    p_dx = get_probable(xs)
    #p_dx_f = get_rep(xs)
    p_dy = get_probable(ys)
    print("Probable grid dx:%f x dy:%f" % (p_dx, p_dy))
    #print("Prob grid(FFT): dx:%f" % p_dx_f)

    def get_phase(vs, delta):
        return np.median([v % delta for v in vs])

    ph_x = get_phase(xs, p_dx)
    ph_y = get_phase(ys, p_dy)
    print("Probable phase px:%f py:%f" % (ph_x, ph_y))

    grid_desc = ((p_dx, ph_x), (p_dy, ph_y))

    if visualize:
        img_debug = cv2.cvtColor(img_gray, cv.CV_GRAY2BGR)
        for ix in range(20):
            x = int(p_dx * ix + ph_x)
            cv2.line(img_debug, (x, 0), (x, 1000), (0, 0, 255), thickness=3)
        for iy in range(20):
            y = int(p_dy * iy + ph_y)
            cv2.line(img_debug, (0, y), (1000, y), (0, 255, 0), thickness=3)
        cv2.imwrite('debug/proc-%s-grid.png' % photo_id, img_debug)

    return (img_depersp, grid_desc, trans_persp)


def detect_lines(img_bin, num_lines_target, n_iterations=5):
    """
    Try to detect specified number of most salient lines in
    img_bin.

    This implementation is not fully generic enough,
    use with 100<=num_lines_target<=1000.

    return lines [(rho, theta)]
    """
    vote_thresh = 500
    change_rate = 0.8
    for i in range(n_iterations):
        lines = cv2.HoughLines(img_bin, 2, 0.01, int(vote_thresh))
        n_lines = 0 if lines is None else len(lines[0])
        if n_lines < num_lines_target * 0.7:
            vote_thresh *= change_rate
        elif n_lines > num_lines_target * 1.3:
            vote_thresh /= change_rate
        else:
            break
        change_rate = change_rate * 0.9 + 0.1
    else:
        print("WARN Target(%d) != Achieved(%d)" % (num_lines_target, n_lines))
    assert(lines is not None)
    return lines[0]


def draw_rhotheta_line(img, line, color):
    l_org, l_dir = rhotheta_to_cartesian(*line)
    p0 = tuple((l_org - l_dir * 2000).astype(int))
    p1 = tuple((l_org + l_dir * 2000).astype(int))
    cv2.line(img, p0, p1, color, lineType=cv.CV_AA)


def extract_patches_raw(ortho_image, xs, ys, margin=0.1, patch_size=80):
    """
    xs, ys: ticks of grid (in increasing order)
    Extract 9^2 square patches from ortho_image.
    """
    assert(len(xs) >= 2)
    assert(len(ys) >= 2)
    patches = {}
    height, width, channels = ortho_image.shape
    for (ix, (x0, x1)) in enumerate(zip(xs, xs[1:])):
        dx = x1 - x0
        x0 = max(0, int(x0 - dx * margin))
        x1 = min(width - 1, int(x1 + dx * margin))
        for (iy, (y0, y1)) in enumerate(zip(ys, ys[1:])):
            dy = y1 - y0
            y0 = max(0, int(y0 - dy * margin))
            y1 = min(height - 1, int(y1 + dy * margin))

            raw_patch_image = ortho_image[y0:y1, x0:x1]
            patches[(ix, iy)] = cv2.resize(raw_patch_image, (patch_size, patch_size))
    return patches


def extract_patches(ortho_image, xs, ys, margin=0.1, patch_size=80):
    """
    xs, ys: ticks of grid (in increasing order)
    Extract 9^2 square patches from ortho_image.
    """
    assert(len(xs) == 10)
    assert(len(ys) == 10)
    patches = {}
    height, width, channels = ortho_image.shape
    for (ix, (x0, x1)) in enumerate(zip(xs, xs[1:])):
        dx = x1 - x0
        x0 = max(0, int(x0 - dx * margin))
        x1 = min(width - 1, int(x1 + dx * margin))
        for (iy, (y0, y1)) in enumerate(zip(ys, ys[1:])):
            dy = y1 - y0
            y0 = max(0, int(y0 - dy * margin))
            y1 = min(height - 1, int(y1 + dy * margin))

            raw_patch_image = ortho_image[y0:y1, x0:x1]
            patches[(10 - (ix + 1), iy + 1)] = {
                "image": cv2.resize(raw_patch_image, (patch_size, patch_size))
            }
    return patches


def extract_patches_by_corners(image, corners, margin=0.1, patch_size=80):
    """
    Extract 9^2 square patches from normal image and 4 corners in
    image.

       patch
    <--------------->
       |         |
       |cell    |
    <->|<------->|<->
    margin        margin
    """
    cell_px = int(patch_size / (1 + margin * 2))
    margin_px = int(cell_px * margin)
    depersp_size = cell_px * 9 + margin_px * 2
    pts_correct = []
    for (ix, iy) in [(1, 0), (0, 0), (0, 1), (1, 1)]:
        pts_correct.append(np.array([
            margin_px + ix * (depersp_size - 2 * margin_px),
            margin_px + iy * (depersp_size - 2 * margin_px)]))

    trans_persp = cv2.getPerspectiveTransform(
        np.array(corners).astype(np.float32),
        np.array(pts_correct).astype(np.float32))
    #if la.det(trans_persp) < 0:
     #   raise RuntimeError("Corners resulted in strange perspective transform")

    img_depersp = cv2.warpPerspective(
        image, trans_persp, (depersp_size, depersp_size),
        borderMode=cv2.BORDER_REPLICATE)

    cv2.imwrite('debug/ep-depersp.png', img_depersp)

    patches = {}
    for ix in range(1, 10):
        for iy in range(1, 10):
            x0 = (9 - ix) * cell_px
            y0 = (iy - 1) * cell_px
            patches[(ix, iy)] = img_depersp[y0:y0+patch_size, x0:x0+patch_size]
    return patches


def to_cairo_surface(img):
    """
    Convert OpenCV BGR image to cairo.ImageSurface
    """
    h, w, c = img.shape
    assert(c == 3)
    # BGR -> BGRX(little endian)
    arr = np.zeros([h, w, 4], np.uint8)
    arr[:, :, :3] = img[:, :, :]
    return cairo.ImageSurface.create_for_data(
        arr, cairo.FORMAT_RGB24, w, h)


def detect_board_region(photo_id, img, img_gray, visualize=False):
    """
    Detect quad region larger than (and parallel to) actual grid.
    return: 4-corners in image space, CCW or None
    """
    img_bin = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5)
    if visualize:
        cv2.imwrite('debug/proc-%s-binary.png' % photo_id, img_bin)
    # Detect lines. None or [[(rho,theta)]]
    lines = detect_lines(img_bin, 30, 10)
    lines_weak = detect_lines(img_bin, 1000)
    if visualize:
        img_gray_w_lines = cv2.cvtColor(img_gray, cv.CV_GRAY2BGR) * 0
        for line in lines:
            draw_rhotheta_line(img_gray_w_lines, line, (0, 0, 255))
        cv2.imwrite('debug/proc-%s-raw-lines.png' % photo_id, img_gray_w_lines)
    # Detect vanishing points
    hfov, vps = detect_board_vps(photo_id, img_gray.shape, lines, visualize)
    # Convert weak lines to normals of great circles.
    focal_px = img_gray.shape[0] / 2 / math.tan(hfov / 2)
    center = np.array([img_gray.shape[1], img_gray.shape[0]]) / 2
    lines_weak_normals = []
    for line in lines_weak:
        l_org, l_dir = rhotheta_to_cartesian(*line)
        p0 = point_to_direction(focal_px, center, l_org - l_dir * 100)
        p1 = point_to_direction(focal_px, center, l_org + l_dir * 100)
        n = np.cross(p0, p1)
        n /= la.norm(n)
        lines_weak_normals.append(n)
    # Classify
    vp0, vp1 = vps
    inliers0 = []
    inliers1 = []
    dist_thresh = 0.01
    for (lorg, ln) in zip(lines_weak, lines_weak_normals):
        dist0 = abs(math.asin(np.dot(ln, vp0)))
        dist1 = abs(math.asin(np.dot(ln, vp1)))
        if dist0 < dist_thresh and dist1 < dist_thresh:
            continue
        if dist0 < dist_thresh:
            inliers0.append(lorg)
        if dist1 < dist_thresh:
            inliers1.append(lorg)
    if visualize:
        img_vps = np.copy(img)
        for (inliers, color) in [(inliers0, (0, 0, 255)), (inliers1, (0, 255, 0))]:
            for line in inliers:
                draw_rhotheta_line(img_vps, line, color)
        cv2.imwrite('debug/proc-%s-vps-weak.png' % photo_id, img_vps)
    # Any pairs of X and Y lines will form a rectangle.
    # (Unless they're really small)
    #      ly0   ly1  (orders vary)
    # lx0   -------
    #       |      |
    # lx1   -------
    dir0 = rhotheta_to_cartesian(*inliers0[0])[1]
    dir1 = rhotheta_to_cartesian(*inliers1[0])[1]
    if abs(np.dot(dir0, dir1)) > 0.5:
        print('Angle too far from orthogonal')
        return None
    margin = 5
    inliers0.sort(key=lambda x: np.dot(rhotheta_to_cartesian(*x)[0], dir1))
    inliers1.sort(key=lambda x: np.dot(rhotheta_to_cartesian(*x)[0], dir0))
    lxs = [inliers0[0], inliers0[-1]]
    lys = [inliers1[0], inliers1[-1]]
    pts_photo = []
    for (ix, iy) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
        pts_photo.append(intersect_lines(lxs[ix], lys[iy]))
    return pts_photo


def detect_board_corners(photo_id, img, visualize=False):
    """
    * photo_id: str
    * img: BGR image

    Take color image and detect 9x9 black grid in shogi board
    It's assumed that shogi board occupies large portion of img.

    return: (4 corners of grid in CCW order) or None
    """
    assert(len(img.shape) == 3)  # y, x, channel
    assert(img.shape[2] == 3)  # channel == 3
    # Resize image to keep height <= max_height.
    max_height = 1000
    if img.shape[0] > max_height:
        height, width = img.shape[:2]
        resize_factor = max_height / height
        new_size = (int(width * resize_factor), max_height)
        img = cv2.resize(img, new_size)
    else:
        resize_factor = 1.0

    # Apply threshold to try to keep only grids
    # Be generous with noise though, because if grid is faint and
    # gone here, it will be impossible to detect grid in later steps.
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    region = detect_board_region(photo_id, img, img_gray, visualize)
    if region is None:
        return None

    grid_pattern = detect_board_grid(
        photo_id, img, img_gray, region, visualize)
    if grid_pattern is None:
        return None

    # Extract patches
    depersp_img, gp, persp_trans = grid_pattern
    depersp_size = depersp_img.shape[0]

    def fp_to_vs(dv, pv):
        vs = [pv + dv * i for i in range(int(depersp_size / dv) + 1)]
        return filter(lambda v: 0 <= v < depersp_size, vs)

    param_path = "params/cell_validness_20x20_mlp.json.bz2"
    classifier_v = classify.CellValidnessClassifier()
    classifier_v.load_parameters(param_path)

    xs = fp_to_vs(*gp[0])
    ys = fp_to_vs(*gp[1])
    # Extract best range
    if len(xs) < 10 or len(ys) < 10:
        print("%s: not enough cells" % photo_id)
        return None

    patches_raw = extract_patches_raw(depersp_img, xs, ys)
    probs_raw = {}
    for (pos, img) in patches_raw.items():
        label, prob = classifier_v.classify(img)
        p_valid = prob if label == "valid" else 1 - prob
        probs_raw[pos] = p_valid

    # TODO: You can use integral image + log to increase speed
    # by x81.
    best_p = 0
    best_offset = None
    for idx in range(len(xs) - 9):
        for idy in range(len(ys) - 9):
            p = 1.0
            for ix in range(9):
                for iy in range(9):
                    p *= probs_raw[(ix + idx, iy + idy)]
            if p > best_p:
                best_p = p
                best_offset = (idx, idy)

    if visualize:
        debug_img = depersp_img.copy()

        surf = to_cairo_surface(debug_img)
        ctx = cairo.Context(surf)
        # validness of individual cells
        for ((ix, iy), prob) in probs_raw.items():
            p0 = gp[0][1] + gp[0][0] * ix
            p1 = gp[1][1] + gp[1][0] * iy
            # blue:valid red:invalid
            ctx.set_source_rgba(1 - prob, 0, prob, 0.3)
            ctx.rectangle(
                p0, p1,
                gp[0][0], gp[1][0])
            ctx.fill()
            ctx.set_source_rgb(0, 0, 0)
            ctx.save()
            ctx.translate(p0, p1 + gp[1][0])
            ctx.scale(2, 2)
            ctx.show_text("%.2f" % prob)
            ctx.restore()
        # best grid
        ctx.set_source_rgb(0, 1, 0)
        ctx.set_line_width(2)
        ctx.rectangle(
            xs[best_offset[0]],
            ys[best_offset[1]],
            xs[best_offset[0] + 9] - xs[best_offset[0]],
            ys[best_offset[1] + 9] - ys[best_offset[1]])
        ctx.stroke()
        ctx.save()
        ctx.translate(xs[best_offset[0]], ys[best_offset[1]])
        ctx.scale(4, 4)
        ctx.show_text("%.2f" % best_p ** (1 / 81))
        ctx.restore()
        surf.write_to_png("debug/%s-validness.png" % photo_id)

    p_valid_grid = best_p ** (1 / 81)
    print("Patch Validness pid=%s p=%f" % (photo_id, p_valid_grid))
    if p_valid_grid < 0.75:
        print("WARN: rejecting due to low validness score")
        return None

    # Recover corners in the original image space.
    corners_depersp = np.array([
        (xs[best_offset[0]], ys[best_offset[1]]),
        (xs[best_offset[0]], ys[best_offset[1] + 9]),
        (xs[best_offset[0] + 9], ys[best_offset[1] + 9]),
        (xs[best_offset[0] + 9], ys[best_offset[1]])
    ])
    corners_org_small = cv2.perspectiveTransform(
        np.array([corners_depersp]), la.inv(persp_trans))[0]
    corners_org = corners_org_small / resize_factor
    return corners_org


def detect_board(photo_id, img, visualize=False, derive=None):
    """
    * photo_id: str
    * img: BGR image
    Take color image and detect 9x9 black grid in shogi board
    It's assumed that shogi board occupies large portion of img.

    return: None in failure
    """
    corners = detect_board_corners(photo_id, img, visualize)
    if corners is None:
        return None

    # Extract patches
    patches = extract_patches_by_corners(img, corners)

    if derive is not None:
        if derive.derive_emptiness:
            derive_empty_vs_nonempty_samples(photo_id, patches)
        if derive.derive_types_up:
            derive_typed_samples(photo_id, patches)
        if derive.derive_validness:
            derive_validness_samples(photo_id, patches, grid_pattern)

    return {
        "corners": corners,
        "patches": patches
    }


def rotate_patches_90deg(patches):
    """
    Rotate patch dictionary as if original image is rotated by 90-degree
    CCW.
    """
    def rot_pos(i, j):
        """
        Board Coordinates:
              i
        9 ... 1
         ------  1 j
         |    | ...
         ------  9
        """
        return (10 - j, i)

    def rot_patch(patch):
        """
        Image coordinates:
         |---->x
         |
        \|/ y
        """
        return {
            "image": patch["image"].transpose([1, 0, 2])[::-1]
        }

    return {
        rot_pos(*pos): rot_patch(patch)
        for (pos, patch) in patches.items()
    }


def is_vertical(patches):
    """
    Given 9^2 patches in intial configuration, decide rotation
    of the board.

    vertical: principle moving direction is up-down
    horizontal: principle moving direction is left-right

    return is_vertical
    """
    param_path = "params/cell_emptiness_20x20.json.bz2"

    # Guess rotation by using empty vs. occupied information.
    initial_state = shogi.get_initial_configuration()
    always_empty, always_occupied = get_rot_invariants_initial()
    classifier_e = classify.CellEmptinessClassifier()
    classifier_e.load_parameters(param_path)
    non_informative = always_empty | always_occupied
    vote_vertical = 0
    vote_horizontal = 0
    for (pos, patch) in patches.items():
        if pos in non_informative:
            continue
        label, prob = classifier_e.classify(patch["image"])
        vert_expect = 'occupied' if pos in initial_state else 'empty'
        if label == vert_expect:
            vote_vertical += 1
        else:
            vote_horizontal += 1
    vertical = vote_vertical > vote_horizontal
    return vertical


def derive_validness_samples(photo_id, patches_ok, grid_pattern):
    depersp_img, xs, ys = grid_pattern
    patches_bad = []

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # off-by-0.5 error
    xs_off = [x + dx * random.uniform(0.3, 0.7) for x in xs]
    ys_off = [y + dy * random.uniform(0.3, 0.7) for y in ys]
    patches_bad.append(extract_patches(depersp_img, xs_off, ys_off))

    # outside
    #extract_patches(depersp_img, )

    for (pos, patch) in patches_ok.items():
        img = patch["image"]
        name = '%s-%d%d-valid' % (photo_id, pos[0], pos[1])
        cv2.imwrite('derived/cells-validness/%s.png' % name, img)

    for (i, patches) in enumerate(patches_bad):
        for (pos, patch) in patches.items():
            img = patch["image"]
            name = '%s-%d-%d%d-invalid' % (photo_id, i, pos[0], pos[1])
            cv2.imwrite('derived/cells-validness/%s.png' % name, img)


def derive_typed_samples(photo_id, patches):
    """
    Depends on: {empty, occupied} classifier
    """
    vertical = is_vertical(patches)
    if not vertical:
        patches = rotate_patches_90deg(patches)

    initial_conf = shogi.get_initial_configuration()
    for (pos, patch) in patches.items():
        img = patch["image"]
        label = "empty"
        if pos in initial_conf:
            label = initial_conf[pos]
            # down -> up
            if pos[1] <= 3:
                img = img[::-1, ::-1]

        name = '%s-%d%d-%s' % (photo_id, pos[0], pos[1], label)
        cv2.imwrite('derived/cells-types-up/%s.png' % name, img)


def derive_empty_vs_nonempty_samples(photo_id, patches):
    """
    Write empty vs. occupied cell images from patches in
    initial configuration.

    Tolerant to 90-degree rotation.
    """
    # Generate empty vs. non-emtpy samples
    always_empty, always_occupied = get_rot_invariants_initial()
    for (pos, patch) in patches.items():
        label = None
        if pos in always_empty:
            label = "empty"
        elif pos in always_occupied:
            label = "occupied"
        else:
            continue

        cv2.imwrite(
            'derived/cells-emptiness/%s-%d%d-%s.png' % (
                photo_id, pos[0], pos[1], label),
            patches[pos]["image"])


def process_image(packed_args):
    db_path, photo_id, args = packed_args
    print(db_path, photo_id)
    if args.debug:
        print('WARN: using fixed seed 0 for debugging')
        random.seed(0)

    try:
        with sqlite3.connect(db_path) as conn:
            image_blob, corners_truth, corners, config_truth, config = conn.execute(
                """select image, corners_truth, corners, config_truth, config
                from photos where id = ?""",
                (photo_id,)).fetchone()
            img = cv2.imdecode(
                np.fromstring(image_blob, np.uint8), cv.CV_LOAD_IMAGE_COLOR)
        print('processing: id=%s shape=%s' % (photo_id, img.shape))

        if args.derive_cells:
            if corners_truth and config_truth:
                print('Extracting')
                corners = json.loads(corners)
                config = json.loads(config)
                patches = extract_patches_by_corners(img, corners)
                for (key, image) in patches.items():
                    key_str = "%d%d" % key
                    cell_st = config[key_str]
                    # direct
                    path = '%d-%s-%s-%s.png' % (
                        photo_id, key_str, cell_st["state"], cell_st["type"])
                    cv2.imwrite(os.path.join('derived/cells', path), image)
                    # flipped
                    flipped_directions = {
                        'empty': 'empty',
                        'up': 'down',
                        'down': 'up'
                    }
                    path_f = '%d-%sF-%s-%s.png' % (
                        photo_id, key_str, flipped_directions[cell_st["state"]], cell_st["type"])
                    image_f = image[::-1, ::-1]
                    cv2.imwrite(os.path.join('derived/cells', path_f), image_f)
                return {
                    "success": 1
                }
            else:
                return {}

        detected = detect_board(
            str(photo_id), img, visualize=args.debug, derive=args)
        if detected is not None:
            with sqlite3.connect(db_path) as conn:
                if args.guess_grid and not corners_truth:
                    print("Writing", detected["corners"])
                    conn.execute(
                        'update photos set corners=? where id = ?',
                        (json.dumps(map(list, detected["corners"])), photo_id))
                    conn.commit()
                if args.guess_config and corners_truth and not config_truth:
                    # TODO: implement patch classification
                    pass

            return {
                "loaded": 1,
                "success": 1
            }
        else:
            return {
                "loaded": 1
            }
            print('->failed')
    except:
        traceback.print_exc()
        return {
            "crash": 1
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
Extract 9x9 cells from photos of shogi board.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset', metavar='DATASET', nargs=1, type=str,
        help='Dataset sqlite path')
    parser.add_argument(
        '-j', nargs='?', metavar='NUM_PROC', type=int, default=1, const=True,
        help='Number of parallel processes')
    parser.add_argument(
        '--derive-emptiness', action='store_true',
        help='Derive emptiness training data')
    parser.add_argument(
        '--derive-types-up', action='store_true',
        help='Derive upright types training data')
    parser.add_argument(
        '--derive-validness', action='store_true',
        help='Derive validness training data')
    parser.add_argument(
        '--derive-cells', action='store_true',
        help='Derive all cell samples for caffe')
    parser.add_argument(
        '--guess-grid', action='store_true',
        help='Guess all grid corners not flagged as ground-truth')
    parser.add_argument(
        '--guess-config', action='store_true',
        help='Guess cell configuration for images with groundtruth grid')
    parser.add_argument(
        '--debug', action='store_true',
        help='Dump debug images to ./debug/. Also fix random.seed.')
    parser.add_argument(
        '--blacklist', nargs='+', type=str, default=[],
        help="Don't process specified photo id")

    args = parser.parse_args()
    assert(args.j >= 1)
    if args.derive_emptiness:
        clean_directory("derived/cells-emptiness")
    if args.derive_types_up:
        clean_directory("derived/cells-types-up")
    if args.derive_validness:
        clean_directory("derived/cells-validness")
    if args.derive_cells:
        clean_directory("derived/cells")

    pid_blacklist = set(args.blacklist)

    db_path = args.dataset[0]
    conn = sqlite3.connect(db_path)
    pids = [row[0] for row in conn.execute('select id from photos').fetchall()]

    pool = multiprocessing.Pool(args.j)
    ls = []
    for pid in pids:
        if pid in pid_blacklist:
            continue
        ls.append((db_path, pid, args))
    # HACK: receive keyboard interrupt correctly
    # https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
    results = pool.map_async(process_image, ls).get(1000)

    count = {}
    for result in results:
        for (k, v) in result.items():
            count[k] = count.get(k, 0) + v
    print(count)

    # Create text list file from directory content (images).
    cell_to_id = {}
    cell_to_id[("empty", "empty")] = 0
    for (i, t) in enumerate(shogi.all_types):
        cell_to_id[("up", t)] = 1 + i
        cell_to_id[("down", t)] = 1 + len(shogi.all_types) + i

    if args.derive_cells:
        ratio_train = 0.8
        ls = []
        for p in os.listdir("derived/cells"):
            c_state, c_type = p.split('.')[-2].split('-')[2:]
            c_id = cell_to_id[(c_state, c_type)]
            ls.append((os.path.join("derived/cells", p), c_id))
        random.shuffle(ls)
        n_train = int(len(ls) * ratio_train)
        with open('derived/cells/train.txt', 'w') as f:
            for entry in ls[:n_train]:
                f.write('%s %d\n' % entry)
        with open('derived/cells/test.txt', 'w') as f:
            for entry in ls[n_train:]:
                f.write('%s %d\n' % entry)
        print('%d training + %d test sampes derived' % (
            n_train, len(ls) - n_train))
