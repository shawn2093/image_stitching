import cv2
import numpy as np
import math
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from numpy.linalg import norm, svd, inv

def sift(img_path):
    img_colors = cv2.imread(img_path)
    img = img_colors.copy()
    img_grey = cv2.imread(img_path, 0)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_grey, None)
    cv2.drawKeypoints(img_colors, keypoints, img_colors, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_name = img_path.split('/')[-1].split('.')[0]
    cv2.imwrite(f"sift_{img_name}.jpg", img_colors)
    return img, keypoints, descriptors, img_name

def feature_match(img_path1, img_path2, ratio, directory):
    img1, kp1, des1, name1 = sift(img_path1)
    img2, kp2, des2, name2 = sift(img_path2)
    size1, size2 = len(des1), len(des2)
    h1, w1 = img1.shape[0], img1.shape[1]
    h2, w2 = img2.shape[0], img2.shape[1]
    diff_matrix = np.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            diff_matrix[i][j] = norm(des1[i] - des2[j])
    matches = []
    for idx in range(size1):
        best2 = np.argsort(diff_matrix[idx])[:2]
        matches.append([idx, best2[0]]) if diff_matrix[idx][best2[0]] < diff_matrix[idx][best2[1]] * ratio else None
    output = np.zeros((max(h1, h2), w1 + w2, 3), dtype= 'uint8')
    output[0:h1, 0:w1] = img1
    output[0:h2, w1:] = img2
    for i, j in matches:
        color = list(map(int, np.random.randint(0, high=255, size=(3,))))
        pts1 = (int(kp1[i].pt[0]), int(kp1[i].pt[1]))
        pts2 = (int(kp2[j].pt[0] + w1), int(kp2[j].pt[1]))
        cv2.line(output, pts1, pts2, color, 1)
    cv2.imwrite(f"{directory}matchline_{name1}_{name2}.jpg", output)
    return matches, kp1, kp2

def homomat(min_match_count: int, src, dst):
    A = np.zeros((min_match_count * 2, 9))
    # construct the two sets of points
    for i in range(min_match_count):
        src1, src2 = src[i, 0, 0], src[i, 0, 1]
        dst1, dst2 = dst[i, 0, 0], dst[i, 0, 1]
        A[i * 2, :] = [src1, src2, 1, 0, 0, 0, -src1 * dst1, - src2 * dst1, -dst1]
        A[i * 2 + 1, :] = [0, 0, 0, src1, src2, 1, -src1 * dst2, - src2 * dst2, -dst2]
    
    # compute the homography between the two sets of points
    [_, S, V] = svd(A)
    m = V[np.argmin(S)]
    m *= 1 / m[-1]
    H = m.reshape((3, 3))
    return H

def ransac(matches, kp1, kp2, min_match_count, num_test:int, threshold: float):
    if len(matches) > min_match_count:
        src_pts = np.array([kp2[m[1]].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.array([kp1[m[0]].pt for m in matches]).reshape(-1, 1, 2)
        min_outliers_count = math.inf
        
        for _ in range(num_test):
            indexs = np.random.choice(len(matches), min_match_count, replace=False)
            homography = homomat(min_match_count, src_pts[indexs], dst_pts[indexs])

            # Warp all left points with computed homography matrix and compare SSDs
            src_pts_reshape = src_pts.reshape(-1, 2)
            one = np.ones((len(src_pts_reshape), 1))
            src_pts_reshape = np.concatenate((src_pts_reshape, one), axis=1)
            warped_left = np.array(np.mat(homography) * np.mat(src_pts_reshape).T)
            for i, value in enumerate(warped_left.T):
                warped_left[:, i] = (value * (1 / value[2])).T

            # Calculate SSD
            dst_pts_reshape = dst_pts.reshape(-1, 2)
            dst_pts_reshape = np.concatenate((dst_pts_reshape, one), axis=1)
            inlier_count = 0
            inlier_list = []
            for i, pair in enumerate(matches):
                ssd = np.linalg.norm(np.array(warped_left[:, i]).ravel() - dst_pts_reshape[i])
                if ssd <= threshold:
                    inlier_count += 1
                    inlier_list.append(pair)

            if (len(matches) - inlier_count) < min_outliers_count:
                min_outliers_count = (len(matches) - inlier_count)
                best_homomat = homography
                best_matches = inlier_list
        return best_homomat, best_matches
    else:
        raise Exception("Not much matching keypoints exist!")

def linear_blend(res, window_size, img1, img2, img_name, direction):
    top = direction[0]
    left = direction[2]
    res = res.copy()
    alpha = step_a = window = step_w = 0
    for m in range(img1.shape[1]):
        alpha += step_a
        window += step_w
        for n in range(img1.shape[0]):
            if sum(res[n+top, m+left]) != 0:
                if window==0:
                    step_w = 1/(img1.shape[1]-m)
                    window+=step_w
                if window > window_size and alpha==0:
                    step_a = 1/(img1.shape[1]-m)
                    alpha+=step_a
                res[n+top, m+left] = alpha * res[n+top, m+left] + (1 - alpha) * img1[n, m]
            else:
                res[n+top, m+left] = img1[n, m]
    cv2.imwrite(f"linear_window_warp_{img_name}.jpg", res)
    return direction, res

def inverse_warp(h, w, corners, homography, img1, img2, img_name, direction):
    
    res_bi = np.zeros((h, w, 3), dtype='uint8')

    top = direction[0]
    left = direction[2]

    # Create image 2 grid in image 1 coordinate
    b, t, r, l = math.ceil(max(corners[:, 1])),math.floor(min(corners[:, 1])),math.ceil(max(corners[:, 0])), math.floor(min(corners[:, 0]))

    img2_trans_grid = [[n, m, 1] for n in range(l, r) for m in range(min(t, 1), b)]
    
    # Inverse mapping points on image 1 to image 2
    img2_trans_inv = np.array(np.mat(inv(homography)) * np.mat(img2_trans_grid).T)
    img2_trans_inv /= img2_trans_inv[2]

    for x, y, im in zip(img2_trans_inv[0], img2_trans_inv[1], img2_trans_grid):
        if math.ceil(y) < img2.shape[0] and math.ceil(y)>0 and math.ceil(x) < img2.shape[1] and math.ceil(x)>0:
            # Bilinear interpolation
            res_bi[im[1]+top,im[0]+left] = (img2[math.ceil(y), math.ceil(x), :]*((y-math.floor(y))*(x-math.floor(x)))+
                                                 img2[math.floor(y), math.floor(x), :]*((math.ceil(y)-y)*(math.ceil(x)-x))+
                                                 img2[math.ceil(y), math.floor(x), :]*((y-math.floor(y))*(math.ceil(x)-x))+
                                                 img2[math.floor(y), math.ceil(x), :]*((math.ceil(y)-y)*(x-math.floor(x))))
    cv2.imwrite(f'backward_bi_{img_name}.jpg', res_bi)
    return linear_blend(res_bi, 0, img1, img2, img_name, direction)

def warp(homography, img1, img2, img_name):
    
    # Transform image 2 with homography matrix
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img2_grid = [[n, m, 1] for n in range(img2.shape[1]) for m in range(img2.shape[0])]
    img2_trans = np.array(np.mat(homography) * np.mat(img2_grid).T)
    img2_trans /= img2_trans[2]

    # Find transformed four corners of image 2 on image 1 coordinate system

    corners = np.zeros((4, 3))
    for p, im in zip(img2_trans.T, img2_grid):
        if im[0] == 0 and im[1] == 0:
            corners[0] = p
        elif im[0] == 0 and im[1] == img2.shape[0] - 1:
            corners[1] = p
        elif im[0] == img2.shape[1] - 1 and im[1] == 0:
            corners[2] = p
        elif im[0] == img2.shape[1] - 1 and im[1] == img2.shape[0] - 1:
            corners[3] = p
            
    # Blended image size
    top = max(0, math.ceil(-min(corners.T[1])))
    bottom = max(img1.shape[0], math.ceil(max(corners.T[1])))+top
    left = max(0, math.ceil(-min(corners.T[0])))
    right = max(img1.shape[1], math.ceil(max(corners.T[0])))+left
    r = max(img1.shape[1], math.ceil(min(corners[2][0],corners[3][0])))+left
    direction = [top, bottom, left, r]
    # Inverse warping
    return inverse_warp(bottom, right, corners, homography, img1, img2, img_name, direction)

def stitching(img1, img2, ratio = 0.5, directory):
    img1_cv = cv2.imread(img1)
    img2_cv = cv2.imread(img2)
    img_name = img1.split('/')[-1].split('.')[0] + '_' + img2.split('/')[-1].split('.')[0]
    match, kp1, kp2 = feature_match(img1, img2, ratio)
    best_homo, best_match = ransac(match, kp1, kp2, 8, 1000, 0.5)
    direction, res_img = warp(best_homo, img1, img2, img_name)
    top = direction[0]
    bottom = direction[0] + img1_cv.shape[0]
    left = direction[2]
    right = direction[3]
    cv2.imwrite(f'{directory}≥…panorama_{img_name}.jpg', res_img[top:bottom, left:right])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--left", type=str, default="./data/1.jpg")
    parser.add_argument("--right", type=str, default="./data/2.jpg")
    parser.add_argument("--output", type=str, default="./")
    parser.add_argument("--ratio", type=float, default=0.5)
    args = parser.parse_args()

    if img1 is None or img2 is None:
        print("Error loading images.")
        exit()
    stitching(args.left, args.right, args.ratio, args.output)
    