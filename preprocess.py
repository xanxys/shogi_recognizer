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
                max_n_inliers = n_inliers
                max_ns = (n0, n1)
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
                max_inliers = (inl0, inl1)
                max_fov = hfov
    print("Max: fov=%.1f #inl=%d axis=%s" % (max_fov, max_n_inliers, max_ns))
    if visualize:
        img_vps = np.copy(img)
        for (ix, color) in [(0, (0, 0, 255)), (1, (0, 255, 0))]:
            for (rho, theta) in max_inliers[ix]:
                l_org, l_dir = rhotheta_to_cartesian(rho, theta)
                p0 = tuple((l_org - l_dir * 2000).astype(int))
                p1 = tuple((l_org + l_dir * 2000).astype(int))
                cv2.line(img_vps, p0, p1, color, lineType=cv.CV_AA)
        cv2.imwrite('debug/%s-vps.png' % photo_id, img_vps)
    return (max_fov, max_ns)


def detect_board_pattern(photo_id, img, lines, lines_weak, visualize):
    """
    Detect shogi board pattern in lines.

    * img: size reference and background for visualization
    * lines: strong lines which is used to initialize VPs
    * lines_weak: lines used to guess final board. must contain all grid lines

    Use "3-line RANSAC" with variable hfov.
    (it operates on surface of a sphere)

    The center of img must be forward direction (i.e. img must not be cropped)
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
            for (rho, theta) in inliers:
                l_org, l_dir = rhotheta_to_cartesian(rho, theta)
                p0 = tuple((l_org - l_dir * 2000).astype(int))
                p1 = tuple((l_org + l_dir * 2000).astype(int))
                cv2.line(img_vps, p0, p1, color, lineType=cv.CV_AA)
        cv2.imwrite('debug/%s-vps-weak.png' % photo_id, img_vps)
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
    inliers0.sort(key=lambda x: np.dot(rhotheta_to_cartesian(*x)[0], dir1))
    inliers1.sort(key=lambda x: np.dot(rhotheta_to_cartesian(*x)[0], dir0))
    lxs = [inliers0[0], inliers0[-1]]
    lys = [inliers1[0], inliers1[-1]]
    pts_photo = []
    pts_correct = []
    for (ix, iy) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
        pts_photo.append(intersect_lines(lxs[ix], lys[iy]))
        pts_correct.append(np.array([ix * depersp_size,iy * depersp_size]))
    trans_persp = cv2.getPerspectiveTransform(
        np.array(pts_photo).astype(np.float32), np.array(pts_correct).astype(np.float32))
    # Correct perspectiveness.
    # This will result in orthogonal image with elongation.
    img_depersp = cv2.warpPerspective(img, trans_persp, (depersp_size, depersp_size))
    if la.det(trans_persp) < 0:
        img_depersp = img_depersp[:, ::-1, :]
    if visualize:
        cv2.imwrite('debug/%s-depersp.png' % photo_id, img_depersp)



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


def detect_board(photo_id, img, visualize):
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
        cv2.imwrite('debug/%s-binary.png' % photo_id, img_bin)
    # Detect lines. None or [[(rho,theta)]]
    lines = detect_lines(img_bin, 30, 10)
    lines_weak = detect_lines(img_bin, 1000)
    if visualize:
        img_gray_w_lines = cv2.cvtColor(img_gray, cv.CV_GRAY2BGR) * 0
        for (rho, theta) in lines:
            l_org, l_dir = rhotheta_to_cartesian(rho, theta)
            p0 = tuple((l_org - l_dir * 2000).astype(int))
            p1 = tuple((l_org + l_dir * 2000).astype(int))
            cv2.line(img_gray_w_lines, p0, p1, (0, 0, 255), lineType=cv.CV_AA)
        cv2.imwrite('debug/%s-raw-lines.png' % photo_id, img_gray_w_lines)

    detect_board_pattern(photo_id, img, lines, lines_weak, visualize)
    return True


def process_image(packed_args):
    photo_id, img_path = packed_args
    print(img_path)
    try:
        img = cv2.imread(img_path)
        print('processing %s: id=%s shape=%s' % (img_path, photo_id, img.shape))
        detected = detect_board(str(photo_id), img, True)
        if not detected:
            print('->failed')
    except:
        traceback.print_exc()


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

    args = parser.parse_args()
    assert(args.j >= 1)

    dir_path = args.dataset[0]
    pool = multiprocessing.Pool(args.j)
    # HACK: receive keyboard interrupt correctly
    # https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
    ls = [(photo_id, os.path.join(dir_path, p)) for (photo_id, p) in enumerate(os.listdir(dir_path))]
    pool.map_async(process_image, ls).get(1000)
