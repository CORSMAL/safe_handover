""" This script projects the keypoints of the hand into the 3 views of Corsmal Containers Manipulation Dataset """

import cv2
import os
import argparse
import numpy as np
import json

from libs.perception.mvg.projection import projection
from libs.perception.tracker import readCalibration


def get_filenames(dataset_dir, extension):
    """ Returns a list of filenames.
    Args:
      dataset_dir: A directory containing a set of files.
      extension: extension of the file to return.
    Returns:
      A list of file paths, relative to `dataset_dir and depending on `extension.
    """

    filenames = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if path.endswith(extension):
            filenames.append(path)

    return filenames


def visualise_keypoints(cap1, cap2, cap3, length, path_to_keypoints):
    """ Visualises the hand keypoints in the three camera views.
    Args:
      cap1: opencv video capture of the first view
      cap2: opencv video capture of the second view
      cap3: opencv video capture of the third view
      length: number of frames of the three videos
      path_to_keypoints: list of paths to keypoint json files ordered per frame
    """
    for i in range(0, length):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret1, frame1 = cap1.read()

        cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret2, frame2 = cap2.read()

        cap3.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret3, frame3 = cap3.read()

        if not ret1 or not ret2 or not ret3:
            continue
        num_hands = 0
        keypoints = []
        with open(path_to_keypoints[i], 'r') as f:
            keypoints_ann = json.load(f)
            if "hand_left_keypoints_3d" in keypoints_ann["people"]:
                keypoints.append(keypoints_ann["people"]["hand_left_keypoints_3d"])
                num_hands += 1
            if "hand_right_keypoints_3d" in keypoints_ann["people"]:
                keypoints.append(keypoints_ann["people"]["hand_right_keypoints_3d"])
                num_hands += 1
        chunk_size = 3
        for j in range(0, num_hands):
            for z in range(0, len(keypoints[j]), chunk_size):
                p3d = np.array(keypoints[j][z:z + chunk_size])
                # Reproject to C1
                p2d_c1, _ = cv2.projectPoints(p3d, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'],
                                              c1.intrinsic['rgb'], np.array([0., 0., 0., 0., 0.]))
                p2d_c1 = p2d_c1.squeeze().astype(int)
                # frame1[p2d_c1[1], p2d_c1[0], :] = [0, 255, 0]
                for j in range(0, len(p2d_c1)):
                    frame1 = cv2.circle(frame1, (p2d_c1[0], p2d_c1[1]), 2, [153, 153, 0], -1)
                if (p2d_c1[0] >= frame1.shape[1]).any() or (p2d_c1[0] < 0).any() or (
                        p2d_c1[1] >= frame1.shape[0]).any() or (p2d_c1[1] < 0).any():
                    continue

                # Reproject to C2
                p2d_c2, _ = cv2.projectPoints(p3d, c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'],
                                              c2.intrinsic['rgb'], np.array([0., 0., 0., 0., 0.]))
                p2d_c2 = p2d_c2.squeeze().astype(int)
                # frame2[p2d_c2[1], p2d_c2[0], :] = [0, 255, 0]
                for j in range(0, len(p2d_c2)):
                    frame2 = cv2.circle(frame2, (p2d_c2[0], p2d_c2[1]), 2, [153, 153, 0], -1)
                if (p2d_c2[0] >= frame2.shape[1]).any() or (p2d_c2[0] < 0).any() or (
                        p2d_c2[1] >= frame2.shape[0]).any() or (p2d_c2[1] < 0).any():
                    continue

                # Reproject to C3
                p2d_c3, _ = cv2.projectPoints(p3d, c3.extrinsic['rgb']['rvec'], c3.extrinsic['rgb']['tvec'],
                                              c3.intrinsic['rgb'], np.array([0., 0., 0., 0., 0.]))
                p2d_c3 = p2d_c3.squeeze().astype(int)
                if (p2d_c3[0] >= frame3.shape[1]).any() or (p2d_c3[0] < 0).any() or (
                        p2d_c3[1] >= frame3.shape[0]).any() or (p2d_c3[1] < 0).any():
                    continue
                for j in range(0, len(p2d_c3)):
                    frame3 = cv2.circle(frame3, (p2d_c3[0], p2d_c3[1]), 2, [153, 153, 0], -1)

        # cv2.imshow("Camera1", frame1)
        # cv2.imshow("Camera2", frame2)
        # cv2.imshow("Camera3", frame3)
        dim = (frame1.shape[1] // 2, frame1.shape[0] // 2)
        cv2.imshow("Cameras", np.hstack([cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA),
                                         cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA),
                                         cv2.resize(frame3, dim, interpolation=cv2.INTER_AREA)]))
        cv2.waitKey(10)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoints_file_dir', type=str,
                        default=".../data/vision_estimations/openpose/1/s0_fi0_fu0_b0_l0")
    parser.add_argument('--calib1_file_path', type=str,
                        default=".../view1/calib/000618.pickle")
    parser.add_argument('--calib2_file_path', type=str,
                        default=".../view2/calib/000618.pickle")
    parser.add_argument('--calib3_file_path', type=str,
                        default=".../view3/calib/000618.pickle")
    parser.add_argument('--video1_file_path', type=str, default=".../view1/rgb/000618.mp4")
    parser.add_argument('--video2_file_path', type=str, default=".../view2/rgb/000618.mp4")
    parser.add_argument('--video3_file_path', type=str, default=".../view3/rgb/000618.mp4")
    args = parser.parse_args()

    # Assert paths exist
    assert os.path.exists(args.keypoints_file_dir), "Path to keypoints annotation directory does not exists"
    assert os.path.exists(args.calib1_file_path), "Path to view 1 calibration file does not exists"
    assert os.path.exists(args.calib2_file_path), "Path to view 2 calibration file does not exists"
    assert os.path.exists(args.calib3_file_path), "Path to view 3 calibration file does not exists"
    assert os.path.exists(args.video1_file_path), "Path to view 1 video file does not exists"
    assert os.path.exists(args.video2_file_path), "Path to view 2 video file does not exists"
    assert os.path.exists(args.video3_file_path), "Path to view 3 video file does not exists"

    # View 1
    v1_calib_path = args.calib1_file_path
    c1 = projection(camId=1)
    c1 = readCalibration(v1_calib_path, c1)
    v1_video = args.video1_file_path
    cap1 = cv2.VideoCapture(v1_video)

    # View 2
    v2_calib_path = args.calib2_file_path
    c2 = projection(camId=2)
    c2 = readCalibration(v2_calib_path, c2)
    v2_video = args.video2_file_path
    cap2 = cv2.VideoCapture(v2_video)

    # View3
    v3_calib_path = args.calib3_file_path
    c3 = projection(camId=1)
    c3 = readCalibration(v3_calib_path, c3)
    v3_video = args.video3_file_path
    cap3 = cv2.VideoCapture(v3_video)

    length1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    length2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    length3 = int(cap3.get(cv2.CAP_PROP_FRAME_COUNT))
    assert length2 == length1 == length3, "Videos have different number of frames"

    path_to_keypoints = get_filenames(args.keypoints_file_dir, ".json")
    path_to_keypoints.sort()

    visualise_keypoints(cap1, cap2, cap3, length1, path_to_keypoints)
