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


def clean_directory(dir_path):
    """
    Delete all files in dir_path.
    """
    for path in os.listdir(dir_path):
        os.unlink(os.path.join(dir_path, path))


def get_initial_configuration():
    """
    Return (pos, type)
    pos: (1, 1) - (9, 9)
    type will be 2-letter strings like CSA format.
    (e.g. "FU", "HI", etc.)
    """
    initial_state_top = {
        (1, 1): "KY",
        (2, 1): "KE",
        (3, 1): "GI",
        (4, 1): "KI",
        (5, 1): "OU",
        (6, 1): "KI",
        (7, 1): "GI",
        (8, 1): "KE",
        (9, 1): "KY",
        (2, 2): "KA",
        (8, 2): "HI",
        (1, 3): "FU",
        (2, 3): "FU",
        (3, 3): "FU",
        (4, 3): "FU",
        (5, 3): "FU",
        (6, 3): "FU",
        (7, 3): "FU",
        (8, 3): "FU",
        (9, 3): "FU",
    }
    initial_state = {}
    for (pos, ty) in initial_state_top.items():
        x, y = pos
        initial_state[pos] = ty
        initial_state[(10 - x, 10 - y)] = ty
    return initial_state


def get_rot_invariants_initial():
    """
    Return positions invariant to 90-degree rotation,
    only considering empty vs occupied categories.

    (always_empty, always_occupied)
    """
    initial_state = get_initial_configuration()
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


def detect_board_vps(photo_id, img, lines, visualize):
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
        focal_px = img.shape[0] / 2 / math.tan(hfov / 2)
        center = np.array([img.shape[1], img.shape[0]]) / 2
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
    if visualize:
        img_vps = np.copy(img)
        for (ix, color) in [(0, (0, 0, 255)), (1, (0, 255, 0))]:
            for line in max_inliers[ix]:
                draw_rhotheta_line(img_vps, line, color)
        cv2.imwrite('debug/proc-%s-vps.png' % photo_id, img_vps)
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


def detect_board_pattern(photo_id, img, lines, lines_weak, visualize):
    """
    Detect shogi board pattern in lines.

    * img: size reference and background for visualization
    * lines: strong lines which is used to initialize VPs
    * lines_weak: lines used to guess final board. must contain all grid lines

    Use "3-line RANSAC" with variable hfov.
    (it operates on surface of a sphere)

    The center of img must be forward direction (i.e. img must not be cropped)

    return (xs, ys) | None
    """
    hfov, vps = detect_board_vps(photo_id, img, lines, visualize)
    # Convert weak lines to normals of great circles.
    focal_px = img.shape[0] / 2 / math.tan(hfov / 2)
    center = np.array([img.shape[1], img.shape[0]]) / 2
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
    depersp_size = 900
    margin = 5
    inliers0.sort(key=lambda x: np.dot(rhotheta_to_cartesian(*x)[0], dir1))
    inliers1.sort(key=lambda x: np.dot(rhotheta_to_cartesian(*x)[0], dir0))
    lxs = [inliers0[0], inliers0[-1]]
    lys = [inliers1[0], inliers1[-1]]
    pts_photo = []
    pts_correct = []
    for (ix, iy) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
        pts_photo.append(intersect_lines(lxs[ix], lys[iy]))
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
    img_gray = cv2.cvtColor(img_depersp, cv.CV_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5)
    thresh = 0.01
    ls = detect_lines(img_bin, 500)
    ls_x = filter(lambda (rho, theta): abs(theta) < thresh, ls)
    ls_y = filter(lambda (rho, theta): abs(theta - math.pi / 2) < thresh, ls)
    print('OrthoLine:%d X:%d Y:%d' % (len(ls), len(ls_x), len(ls_y)))
    if visualize:
        img_debug = cv2.cvtColor(img_gray, cv.CV_GRAY2BGR)
        for line in ls_x:
            draw_rhotheta_line(img_debug, line, (0, 0, 255))
        for line in ls_y:
            draw_rhotheta_line(img_debug, line, (0, 255, 0))
        cv2.imwrite('debug/proc-%s-ortho.png' % photo_id, img_debug)
    if len(ls_x) < 10 or len(ls_y) < 10:
        print('WARN: not enough XY lines')
        return None
    # Detect repetition in each axis
    min_dx = (depersp_size - margin * 2) / 2 / 9  # assume at least half of the image is covered by board
    max_dx = (depersp_size - margin * 2) / 9
    xs = map(lambda line: rhotheta_to_cartesian(*line)[0][0], ls_x)
    ys = map(lambda line: rhotheta_to_cartesian(*line)[0][1], ls_y)
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


    xs = find_9segments(xs, (min_dx, max_dx))
    ys = find_9segments(ys, (min_dx, max_dx))
    print("Lattice candidates: %dx%d" % (len(xs), len(ys)))
    if len(xs) == 0 or len(ys) == 0:
        print("WARN: Couldn't find grid for X or Y")
        return None
    xs = xs[0]
    ys = ys[0]
    if visualize:
        img_debug = cv2.cvtColor(img_gray, cv.CV_GRAY2BGR)
        for x in xs:
            x = int(x)
            cv2.line(img_debug, (x, 0), (x, 1000), (0, 0, 255), thickness=3)
        for y in ys:
            y = int(y)
            cv2.line(img_debug, (0, y), (1000, y), (0, 255, 0), thickness=3)
        cv2.imwrite('debug/proc-%s-grid.png' % photo_id, img_debug)

    if len(xs) == 10 and len(ys) == 10:
        return (img_depersp, xs, ys)
    else:
        return None


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


def detect_board(photo_id, img, visualize=False, derive=None):
    """
    * photo_id: str
    * img: BGR image
    Take color image and detect 9x9 black grid in shogi board
    It's assumed that shogi board occupies large portion of img.

    return: False failure
    """
    assert(len(img.shape) == 3)  # y, x, channel
    assert(img.shape[2] == 3)  # channel == 3
    # Resize image to keep height <= max_height.
    max_height = 1000
    if img.shape[0] > max_height:
        height, width = img.shape[:2]
        new_size = (int(width * max_height / height), max_height)
        img = cv2.resize(img, new_size)
    # Apply threshold to try to keep only grids
    # Be generous with noise though, because if grid is faint and
    # gone here, it will be impossible to detect grid in later steps.
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
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

    grid_pattern = detect_board_pattern(photo_id, img, lines, lines_weak, visualize)
    if grid_pattern is None:
        return False

    # Extract patches
    depersp_img, xs, ys = grid_pattern
    patches = extract_patches(depersp_img, xs, ys)

    # Check patch validness
    param_path = "params/cell_validness_20x20_mlp.json.bz2"
    classifier_v = classify.CellValidnessClassifier()
    classifier_v.load_parameters(param_path)
    p_valid = 1.0
    for patch in patches.values():
        label, prob = classifier_v.classify(patch["image"])
        if label == "valid":
            p_valid *= prob
        else:
            p_valid *= 1 - prob
    p_valid = p_valid ** (1 / len(patches))
    print("Patch Validness pid=%s p=%f" % (photo_id, p_valid))
    if p_valid < 0.8:
        print("WARN: rejecting due to low validness score")
        return False

    if visualize:
        for (pos, patch) in patches.items():
            cv2.imwrite(
                "debug/patch-%s-%d%d.png" % (photo_id, pos[0], pos[1]),
                patch["image"])

    if derive is not None:
        if derive.derive_emptiness:
            derive_empty_vs_nonempty_samples(photo_id, patches)
        if derive.derive_types_up:
            derive_typed_samples(photo_id, patches)
        if derive.derive_validness:
            derive_validness_samples(photo_id, patches, grid_pattern)

    return True


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
    initial_state = get_initial_configuration()
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

    initial_conf = get_initial_configuration()
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
    photo_id, img_path, args = packed_args
    print(img_path)
    if args.debug:
        print('WARN: using fixed seed 0 for debugging')
        random.seed(0)

    try:
        img = cv2.imread(img_path)
        print('processing %s: id=%s shape=%s' % (img_path, photo_id, img.shape))
        detected = detect_board(str(photo_id), img, visualize=args.debug, derive=args)
        if not detected:
            return {
                "loaded": 1
            }
            print('->failed')
        else:
            return {
                "loaded": 1,
                "success": 1
            }
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
        help='Dataset directory path')
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

    pid_blacklist = set(args.blacklist)

    dir_path = args.dataset[0]
    pool = multiprocessing.Pool(args.j)
    ls = []
    for p in os.listdir(dir_path):
        pid = os.path.splitext(p)[0]
        if pid in pid_blacklist:
            continue
        ls.append((pid, os.path.join(dir_path, p), args))
    # HACK: receive keyboard interrupt correctly
    # https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
    results = pool.map_async(process_image, ls).get(1000)

    count = {}
    for result in results:
        for (k, v) in result.items():
            count[k] = count.get(k, 0) + v
    print(count)
