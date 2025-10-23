import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from typing import Iterable, Optional, Sequence

from loguru import logger
from scipy.spatial.distance import cdist

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(
        self,
        tlwh,
        score,
        feat=None,
        feat_history=50,
        activation_wait=30,
        pre_activation_frames=5,
    ):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        self.activation_wait = max(0, activation_wait)
        self.pre_activation_frames = max(0, pre_activation_frames)
        self.public_track_id = -1
        self.pre_activation_features = deque([], maxlen=self.pre_activation_frames)

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        self._maybe_cache_pre_activation_feature(feat)

    def _maybe_cache_pre_activation_feature(self, feat):
        if self.public_track_id != -1:
            return
        if self.pre_activation_frames == 0:
            return
        if self.tracklet_len >= self.activation_wait:
            self.pre_activation_features.append(feat)
            return
        # Capture the last N frames before activation
        frames_required = max(0, self.activation_wait - self.pre_activation_frames)
        if self.tracklet_len > frames_required:
            self.pre_activation_features.append(feat)

    def get_pre_activation_feature(self):
        if self.pre_activation_features:
            stacked = np.stack(self.pre_activation_features, axis=0)
            feature = np.mean(stacked, axis=0)
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature /= norm
            return feature
        return self.smooth_feat

    def assign_public_id(self, new_id: int):
        self.public_track_id = new_id
        if self.public_track_id != -1:
            # Once the track is activated, keep only the aggregated descriptor
            feature = self.get_pre_activation_feature()
            if feature is not None:
                self.smooth_feat = feature
                self.curr_feat = feature
        self.pre_activation_features.clear()

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

    @property
    def display_id(self):
        return self.public_track_id


class BoTSORT(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args
        self.frame_rate = frame_rate

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.activation_wait = getattr(args, "activation_wait", 30)
        self.pre_activation_frames = getattr(args, "pre_activation_frames", 5)
        lost_buffer_time = getattr(args, "lost_track_buffer_time", 300)
        self.max_time_lost = max(self.buffer_size, int(frame_rate * lost_buffer_time))
        self.overlap_reid_iou = getattr(args, "reid_overlap_iou", 0.6)
        self.reid_usage_count = 0
        self.reid_comparisons = 0

        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

        logger.info(
            "BoTSORT configured with activation_wait=%d frames, pre_activation_frames=%d, lost_track_buffer=%d frames",
            self.activation_wait,
            self.pre_activation_frames,
            self.max_time_lost,
        )

    # ------------------------------------------------------------------
    # ReID helpers
    # ------------------------------------------------------------------
    def _should_use_reid_for_association(self, tracks: Sequence[STrack], detections: Sequence[STrack], dets: np.ndarray) -> bool:
        if not self.args.with_reid:
            return False
        if len(tracks) == 0 or len(detections) == 0:
            return False
        if dets is None or len(dets) == 0:
            return False
        track_boxes = [track.tlbr for track in tracks]
        overlaps = matching.ious(track_boxes, dets)
        if overlaps.size == 0:
            return False
        overlap_mask = overlaps > self.overlap_reid_iou
        if np.any(overlap_mask.sum(axis=0) > 1) or np.any(overlap_mask.sum(axis=1) > 1):
            logger.debug("ReID triggered due to overlap ambiguity among %d tracks and %d detections", len(tracks), len(detections))
            return True
        return False

    def _ensure_detection_features(self, img: np.ndarray, detections: Sequence[STrack], indices: Iterable[int], reason: str) -> None:
        if not self.args.with_reid:
            return
        indices = [idx for idx in indices if 0 <= idx < len(detections) and detections[idx].curr_feat is None]
        if not indices:
            return
        boxes = np.asarray([detections[idx].tlbr for idx in indices])
        features = self.encoder.inference(img, boxes)
        if len(features) != len(indices):
            logger.warning("Unexpected number of features (%d) for detections (%d) when running ReID", len(features), len(indices))
            return
        self.reid_usage_count += 1
        logger.info(
            "ReID inference #%d (%s) for %d detections",
            self.reid_usage_count,
            reason,
            len(indices),
        )
        for det_idx, feat in zip(indices, features):
            detections[det_idx].update_features(feat)

    def _ensure_single_detection_feature(self, img: np.ndarray, detection: STrack, reason: str) -> None:
        if not self.args.with_reid or detection.curr_feat is not None:
            return
        self._ensure_detection_features(img, [detection], [0], reason)

    def _ensure_pre_activation_feature(self, img: np.ndarray, track: STrack, detection: STrack) -> None:
        if not self.args.with_reid:
            return
        if track.public_track_id != -1:
            return
        frames_required = max(0, track.activation_wait - track.pre_activation_frames)
        if (track.tracklet_len + 1) <= frames_required:
            return
        if detection.curr_feat is None:
            self._ensure_single_detection_feature(img, detection, "pre-activation feature")
        if detection.curr_feat is not None:
            track._maybe_cache_pre_activation_feature(detection.curr_feat)

    def _match_with_lost_tracks(self, track: STrack) -> Optional[STrack]:
        if not self.lost_stracks:
            logger.debug("Skipping lost-track comparison: no lost tracks to evaluate")
            return None
        feature = track.get_pre_activation_feature()
        if feature is None:
            logger.debug("Skipping lost-track comparison: track lacks appearance features")
            return None
        candidates = [t for t in self.lost_stracks if getattr(t, "smooth_feat", None) is not None]
        if not candidates:
            self.reid_comparisons += 1
            logger.info(
                "Appearance comparison #%d skipped: no lost track features available",
                self.reid_comparisons,
            )
            return None
        lost_features = np.asarray([t.smooth_feat for t in candidates])
        feature = feature.reshape(1, -1)
        distances = cdist(lost_features, feature, metric="cosine").reshape(-1)
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        if min_distance <= self.appearance_thresh:
            matched = candidates[min_idx]
            self.reid_comparisons += 1
            logger.info(
                "Appearance comparison #%d matched lost track %s (distance %.4f) with pending track",
                self.reid_comparisons,
                getattr(matched, "display_id", matched.track_id),
                float(min_distance),
            )
            return matched
        self.reid_comparisons += 1
        logger.info(
            "Appearance comparison #%d found no match (best distance %.4f)",
            self.reid_comparisons,
            float(min_distance),
        )
        return None

    def _assign_track_identity(self, track: STrack) -> None:
        if track.display_id != -1:
            return
        if track.tracklet_len < track.activation_wait:
            return
        matched_lost = self._match_with_lost_tracks(track)
        if matched_lost is not None:
            track.assign_public_id(matched_lost.display_id if matched_lost.display_id != -1 else matched_lost.track_id)
            if matched_lost in self.lost_stracks:
                self.lost_stracks.remove(matched_lost)
            matched_lost.mark_removed()
            self.removed_stracks.append(matched_lost)
            logger.info("Re-activated lost track id=%s", track.display_id)
        else:
            track.assign_public_id(track.track_id)
            logger.info("Assigned new track id=%s after wait of %d frames", track.display_id, track.activation_wait)

    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        if len(dets) > 0:
            '''Detections'''
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    s,
                    activation_wait=self.activation_wait,
                    pre_activation_frames=self.pre_activation_frames,
                )
                for (tlbr, s) in zip(dets, scores_keep)
            ]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        use_reid_association = self._should_use_reid_for_association(strack_pool, detections, dets)
        if use_reid_association and any(track.smooth_feat is None for track in strack_pool):
            logger.debug("Skipping ReID association because some tracks lack appearance features")
            use_reid_association = False

        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid and use_reid_association:
            self._ensure_detection_features(img, detections, range(len(detections)), "overlap association")
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            self._ensure_pre_activation_feature(img, track, det)
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            self._assign_track_identity(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    s,
                    activation_wait=self.activation_wait,
                    pre_activation_frames=self.pre_activation_frames,
                )
                for (tlbr, s) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dets_second_np = np.asarray([det.tlbr for det in detections_second]) if detections_second else np.zeros((0, 4))
        use_reid_second = self._should_use_reid_for_association(r_tracked_stracks, detections_second, dets_second_np)
        if use_reid_second and any(track.smooth_feat is None for track in r_tracked_stracks):
            use_reid_second = False
        if self.args.with_reid and use_reid_second:
            self._ensure_detection_features(img, detections_second, range(len(detections_second)), "second association")
            emb_second = matching.embedding_distance(r_tracked_stracks, detections_second) / 2.0
            emb_second[emb_second > self.appearance_thresh] = 1.0
            iou_mask_second = (dists > self.proximity_thresh)
            emb_second[iou_mask_second] = 1.0
            dists = np.minimum(dists, emb_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            self._ensure_pre_activation_feature(img, track, det)
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            self._assign_track_identity(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        dets_unconfirmed = np.asarray([det.tlbr for det in detections]) if detections else np.zeros((0, 4))
        use_reid_unconfirmed = self._should_use_reid_for_association(unconfirmed, detections, dets_unconfirmed)
        if use_reid_unconfirmed and any(track.smooth_feat is None for track in unconfirmed):
            use_reid_unconfirmed = False

        if self.args.with_reid and use_reid_unconfirmed:
            self._ensure_detection_features(img, detections, range(len(detections)), "unconfirmed association")
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections[idet]
            self._ensure_pre_activation_feature(img, track, det)
            track.update(det, self.frame_id)
            activated_starcks.append(track)
            self._assign_track_identity(track)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
