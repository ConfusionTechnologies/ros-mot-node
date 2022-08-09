import numpy as np
from norfair import Detection
from norfair.tracker import TrackedObject

# TODO: possible to swap out distance functions? like using manhattan or cosine?
# btw if you wanted to insert image feature vector association (like deepSORT), it would be here.
# TODO: feature vector reID


def create_kp_vote_calculator(dist_threshold=1 / 40, conf_threshold=0.5):
    """gauge detection distance by number of keypoints under a threshold of euclidean distance (aka "matching")"""

    def match_distance(pose: Detection, pose_track: TrackedObject):
        dists = np.linalg.norm(pose.points - pose_track.estimate, axis=1)
        num_match = np.count_nonzero(
            (dists < dist_threshold)
            * (pose.scores > conf_threshold)
            * (pose_track.last_detection.scores > conf_threshold)
        )
        return 1 / max(num_match, 1)

    return match_distance


def create_kp_dist_calculator(conf_threshold=0.5, exponent_penalty=1):
    """gauge detection distance by average keypoint distance"""

    def kp_distance(pose: Detection, pose_track: TrackedObject):
        dists = np.linalg.norm(pose.points - pose_track.estimate, axis=1)
        avg_dist = np.mean(
            (
                dists
                * (pose.scores > conf_threshold)
                * (pose_track.last_detection.scores > conf_threshold)
            )
            ** exponent_penalty
        )
        return avg_dist

    return kp_distance


def create_iou_calculator():
    def iou(detection: Detection, tracked_object: TrackedObject):

        x1, y1 = detection.points[0]
        x2, y2 = detection.points[1]
        x3, y3 = tracked_object.estimate[0]
        x4, y4 = tracked_object.estimate[1]

        x_a = max(x1, x3)
        y_a = max(y1, y3)
        x_b = min(x2, x4)
        y_b = min(y2, y4)

        # Compute the area of intersection rectangle
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        # Compute the area of both the prediction and tracker rectangles
        box_a_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box_b_area = (x4 - x3 + 1) * (y4 - y3 + 1)

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + tracker
        # areas - the interesection area
        iou = inter_area / (box_a_area + box_b_area - inter_area)
        # Since 0 <= IoU <= 1, we define 1/IoU as a distance.
        # Distance values will be in [1, inf)
        return 1 - iou

    return iou
