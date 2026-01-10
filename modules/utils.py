import cv2
import numpy
import numpy as np
import pyautogui
import torch
from sklearn.cluster import DBSCAN

from modules.data import im_transform


def auto_eps(values, k=3, factor=1.8, min_eps=1e-3):
    """
    自动估计 DBSCAN eps：
    - 计算每个点到第 k 近邻的距离
    - 取中位数 * factor 作为 eps
    - 确保 eps >= min_eps
    """
    if len(values) < k + 1:
        return min_eps  # 太少直接返回最小值
    values = np.array(values).reshape(-1, 1)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k, len(values) - 1)).fit(values)
    distances, _ = nbrs.kneighbors(values)
    kth_distances = distances[:, -1]
    eps = np.median(kth_distances) * factor
    return max(eps, min_eps)


def cluster_centers(values):
    """对 values（x 或 y）做 DBSCAN 聚类，返回排序后的中心值"""
    eps = auto_eps(values)
    clustering = DBSCAN(eps=5.0, min_samples=2).fit(values.reshape(-1, 1))
    labels = clustering.labels_

    centers = []
    for lab in set(labels):
        if lab == -1:  # -1 是噪声点，忽略
            continue
        cluster_vals = [values[i] for i in range(len(values)) if labels[i] == lab]
        centers.append(np.mean(cluster_vals))

    return sorted(centers)


def points_to_matrix(points):
    """
    ✅ 最稳定版本：基于 DBSCAN 自动分行分列，无需任何参数
    """
    if not points:
        return [], [], []

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    vals = [p[2] for p in points]

    # 1️⃣ 聚类出行中心（y方向）
    row_centers = cluster_centers(ys)

    # 2️⃣ 聚类出列中心（x方向）
    col_centers = cluster_centers(xs)

    # 3️⃣ 初始化矩阵
    R, C = len(row_centers), len(col_centers)
    matrix = np.full((R, C), None)
    dots_matrix = np.zeros((R, C, 2), dtype=int)

    # 4️⃣ 把点映射到最近的(row, col)
    for (x, y, v) in points:
        r = np.argmin([abs(y - rc) for rc in row_centers])
        c = np.argmin([abs(x - cc) for cc in col_centers])
        # 避免冲突 —— 只保留更靠近中心的
        if matrix[r][c] is None:
            matrix[r][c] = v
            dots_matrix[r][c] = [x, y]
        else:
            existing_center_dist = abs(x - col_centers[c]) + abs(y - row_centers[r])
            new_dist = abs(x - col_centers[c]) + abs(y - row_centers[r])
            if new_dist < existing_center_dist:
                matrix[r][c] = v
                dots_matrix[r][c] = [x, y]

    return matrix, dots_matrix



def generate_mask_sizes(W_N,H_N):
    masks_sizes = []
    for mask_h in range(1, H_N + 1):
        for mask_w in range(1, W_N + 1):
            masks_sizes.append((mask_h, mask_w))
    masks_sizes = sorted(masks_sizes, key=lambda x: x[0] * x[1])
    masks_sizes = np.array(masks_sizes)
    return masks_sizes



def revert_to_matrix(image, xycenter, digitals, template_digit_matrix, template_dots_matrix):
    h, w = image.shape[:2]
    xycenter_norm = xycenter.copy().astype(float)
    xycenter_norm[:, 0] = xycenter[:, 0] / w  # x / 宽
    xycenter_norm[:, 1] = xycenter[:, 1] / h  # y / 高
    _matrix = np.zeros_like(template_digit_matrix)
    _positions = np.zeros_like(template_dots_matrix).astype(int)
    for (xn, yn), (x, y), d in zip(xycenter_norm, xycenter, digitals):
        distances = np.linalg.norm(template_dots_matrix - np.array([xn, yn]), axis=2)
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        _matrix[i, j] = int(d)
        _positions[i, j] = [x, y]
    return _matrix, _positions


def digital_recognition(detmodel, image, device):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 250, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True).shape[0] == 4]

    react_xywh = np.array([cv2.boundingRect(contour) for contour in contours])
    dots = react_xywh[:, 2] * react_xywh[:, 3]  # w * h
    avg_dot = numpy.mean(dots)

    keep_dot_indexes = numpy.where((dots > avg_dot * 0.5) & (dots < avg_dot * 1.5))[0]
    react_xywh = react_xywh[keep_dot_indexes]

    react_xywh[:, 0] = react_xywh[:, 0] + react_xywh[:, 2] * 0.1 / 2
    react_xywh[:, 1] = react_xywh[:, 1] + react_xywh[:, 3] * 0.1 / 2
    react_xywh[:, 2] = react_xywh[:, 2] * 0.9
    react_xywh[:, 3] = react_xywh[:, 3] * 0.9

    grid_imgs = [binary[y:y + h, x:x + w] for (x, y, w, h) in react_xywh]

    digits = []

    preim= [im_transform(img) for img in grid_imgs]

    input_tensor = torch.stack(preim).to(device)

    with torch.no_grad():
        outputs = detmodel(input_tensor)
        _, predicted = torch.max(outputs, 1)
        digits = [str(d.item() + 1) for d in predicted]  # 因为标签是从0开始的，所以加1
    return react_xywh,digits


def find_template_region(img_rgb, tpl_rgb, threshold: float = 0.5):
    orb = cv2.ORB_create()

    # 计算图像和模板的关键点及描述符
    kp_img, des_img = orb.detectAndCompute(img_rgb, None)
    kp_tpl, des_tpl = orb.detectAndCompute(tpl_rgb, None)

    # 创建匹配器并进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_tpl, des_img)

    # 根据距离排序匹配结果
    matches = sorted(matches, key=lambda x: x.distance)

    # 选择匹配距离较小的作为有效匹配
    good_matches = [m for m in matches if m.distance < threshold * matches[0].distance]

    if len(good_matches) < 4:
        return None  # 匹配数太少，认为没有找到

    # 计算模板图像和图像中的匹配点之间的透视变换
    src_pts = np.float32([kp_tpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算透视变换矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 获取模板的四个角点
    h, w = tpl_rgb.shape[:2]
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 在输出中返回模板匹配的矩形框区域以及阈值图
    thresh = np.zeros_like(img_rgb)  # 在这里我们返回一个全零的图像（可以根据实际需求更改）

    # 返回位置和阈值图
    x, y, w, h = cv2.boundingRect(dst)
    return (x, y, w, h, thresh)



def drag_mouse(pp, left, top):
    x1, y1 = pp[0][0] + left, pp[0][1] + top
    x2, y2 = pp[1][0] + left, pp[1][1] + top

    pyautogui.moveTo(x1, y1, duration=0.1)
    pyautogui.mouseDown()
    pyautogui.moveTo(x2, y2, duration=0.2)
    pyautogui.mouseUp()


def click_mouse(x, y, left, top):
    px, py = x + left, y + top
    pyautogui.moveTo(px, py, duration=0.1)
    pyautogui.click()


def convert_dot2pixel(bord_image, positions, template_position_matrix, W=10, H=16):
    """
    Convert rects (r1, c1, r2, c2) from solver into pixel pairs using template_position_matrix.
    Solver rect order is row-major; map to (y, x) before lookup.
    """
    if template_position_matrix is None:
        raise ValueError("Template position matrix is not initialized.")

    bw, bh = bord_image.shape[1], bord_image.shape[0]
    tpm = template_position_matrix

    is_transposed = tpm.shape[0] == W and tpm.shape[1] == H
    if is_transposed:
        max_x, max_y = tpm.shape[:2]
    else:
        max_y, max_x = tpm.shape[:2]

    dotps = []
    for pos in positions:
        if not (isinstance(pos, (list, tuple)) and len(pos) == 4):
            continue
        r1, c1, r2, c2 = map(int, pos)  # row/col ordering from backend
        x1, y1, x2, y2 = c1, r1, c2, r2  # convert to x/y for lookup

        in_bounds = (
            0 <= x1 < max_x and 0 <= y1 < max_y and
            0 <= x2 < max_x and 0 <= y2 < max_y
        )
        if not in_bounds:
            continue

        if is_transposed:
            pt1 = tpm[x1][y1].tolist()
            pt2 = tpm[x2][y2].tolist()
        else:
            pt1 = tpm[y1][x1].tolist()
            pt2 = tpm[y2][x2].tolist()

        pt1 = [int(pt1[0] * bw), int(pt1[1] * bh)]
        pt2 = [int(pt2[0] * bw), int(pt2[1] * bh)]
        dotps.append((pt1, pt2))

    return dotps


import requests
def find_best_steps(board, max_times=20, url="http://127.0.0.1:8000/search"):
    payload = {"board": board, "max_time": max_times}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()