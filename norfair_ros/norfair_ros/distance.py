import numpy as np
from norfair import Detection
from norfair.tracker import TrackedObject

# TODO: possible to swap out distance functions? like using manhattan or cosine?
# btw if you wanted to insert image feature vector association (like deepSORT), it would be here.
# TODO: feature vector association


def create_kp_match_calculator(dist_threshold=1 / 40, conf_threshold=0.5):
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
