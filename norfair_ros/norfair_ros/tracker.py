from __future__ import annotations
from dataclasses import dataclass, field
import sys
import random
from colorsys import hsv_to_rgb

import rclpy
from rclpy.node import Node
from ros2topic.api import get_msg_class

import numpy as np
from norfair import Tracker, Detection, OptimizedKalmanFilterFactory

from foxglove_msgs.msg import ImageMarkerArray
from visualization_msgs.msg import ImageMarker
from geometry_msgs.msg import Point
from nicepynode import Job, JobCfg
from nicepynode.utils import declare_parameters_from_dataclass
from norfair_ros.distance import create_kp_dist_calculator

NODE_NAME = "norfair_mot"


@dataclass
class NorfairCfg(JobCfg):
    """"""

    # kp_dist_threshold: float = 1 / 25
    """maximum normalized distance deviation to be considered tracked"""
    kp_score_threshold: float = 0.2
    """minimum confidence to consider a keypoint"""
    score_threshold: float = 0.01  # 0.8
    """overall threshold above which to consider candidates as separate tracks"""
    dets_in_topic: str = "~/dets_in"
    """KEYPOINTS MUST BE NORMALIZED BEFOREHAND!!!!"""
    det_msg_name: str = "poses"
    tracks_out_topic: str = "~/tracks_out"
    markers_out_topic: str = "~/markers"


@dataclass
class NorfairTracker(Job[NorfairCfg]):

    ini_cfg: NorfairCfg = field(default_factory=NorfairCfg)

    def attach_params(self, node: Node, cfg: NorfairCfg):
        super(NorfairTracker, self).attach_params(node, cfg)

        declare_parameters_from_dataclass(node, cfg)

    def attach_behaviour(self, node: Node, cfg: NorfairCfg):
        super(NorfairTracker, self).attach_behaviour(node, cfg)

        # https://github.com/tryolabs/norfair/tree/master/docs#arguments
        self.tracker = Tracker(
            distance_function=create_kp_dist_calculator(cfg.kp_score_threshold),
            # max dist from closest track before candidate is considered as new track
            distance_threshold=cfg.score_threshold,
            # threshold below which keypoint is ignored
            detection_threshold=cfg.kp_score_threshold,
            hit_counter_max=5,  # max HP of track
            initialization_delay=2,  # min HP for track to be considered
            pointwise_hit_counter_max=5,  # max HP of keypoints
            past_detections_length=0,
            filter_factory=OptimizedKalmanFilterFactory(),
        )

        self.log.info(f"Waiting for publisher@{cfg.dets_in_topic}...")
        self.msg_type = get_msg_class(node, cfg.dets_in_topic, blocking=True)
        self._dets_sub = node.create_subscription(
            self.msg_type, cfg.dets_in_topic, self._on_input, 5
        )
        self._track_pub = node.create_publisher(self.msg_type, cfg.tracks_out_topic, 5)
        self._marker_pub = node.create_publisher(
            ImageMarkerArray, cfg.markers_out_topic, 5
        )

        self._id_color_map = {}
        """Used to assign color for visualizing lmao"""

        self.log.info("Ready")

    def detach_behaviour(self, node: Node):
        super().detach_behaviour(node)

        node.destroy_subscription(self._dets_sub)
        node.destroy_publisher(self._track_pub)
        node.destroy_publisher(self._marker_pub)

    def on_params_change(self, node: Node, changes: dict):
        self.log.info(f"Config changed: {changes}.")
        # if not all(n in ("crop_pad",) for n in changes):
        #     self.log.info(f"Config change requires restart.")
        #     return True
        return True

    def _on_input(self, detsmsg):
        # Tracker should run every frame to be accurate
        # if (
        #     self._track_pub.get_subscription_count()
        #     + self._marker_pub.get_subscription_count()
        #     < 1
        # ):
        #     return

        dets = getattr(detsmsg, self.cfg.det_msg_name)

        norfair_dets = tuple(
            Detection(
                points=np.array((d.track.x, d.track.y)).T,
                scores=np.frombuffer(d.track.scores, dtype=np.float32),
                data=d,
                label=d.track.label,
            )
            for d in dets
        )

        # TODO: use time delta for period.
        tracks = self.tracker.update(detections=norfair_dets, period=1)

        tracked_dets = []
        for t in tracks:
            det = t.last_detection.data
            # at no point is det copied
            # so setting here will set it in the original detsmsg
            # unless this happens to be stale (aka last frame's) det...
            # TODO: detect and filter out stale? Or use the Kalman prediction? How to update the timestamp??
            det.track.id = t.id
            tracked_dets.append(det)

        if self._track_pub.get_subscription_count() > 0:
            # republish the msg but with track.id set for each det
            setattr(detsmsg, self.cfg.det_msg_name, tracked_dets)
            self._track_pub.publish(detsmsg)

        if self._marker_pub.get_subscription_count() > 0:
            markersmsg = ImageMarkerArray()

            for d in dets:
                c = self._id_color_map.get(d.track.id, None)
                if c is None:
                    c = self._id_color_map[d.track.id] = random.random()

                color = hsv_to_rgb(c, 1, 1)
                marker = ImageMarker(header=detsmsg.header)
                marker.scale = 4.0
                marker.type = ImageMarker.POINTS
                marker.outline_color.r = float(color[0])
                marker.outline_color.g = float(color[1])
                marker.outline_color.b = float(color[2])
                marker.outline_color.a = 1.0

                # TODO: generalize (dont hardcode), also cannot use track.keypoints as those are normalized...
                marker.points.append(Point(x=float(d.x[0]), y=float(d.y[0])))

                markersmsg.markers.append(marker)

            self._marker_pub.publish(markersmsg)


def main(args=None):
    if __name__ == "__main__" and args is None:
        args = sys.argv

    try:
        rclpy.init(args=args)

        node = Node(NODE_NAME)

        cfg = NorfairCfg()
        NorfairTracker(node, cfg)

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
