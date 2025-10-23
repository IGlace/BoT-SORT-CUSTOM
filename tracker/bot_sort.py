import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from loguru import logger

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    pending_id_counter = 0

    @classmethod
    def next_pending_id(cls):
        cls.pending_id_counter -= 1
        return cls.pending_id_counter

    @classmethod
    def reset_pending_count(cls):
        cls.pending_id_counter = 0

    def __init__(self, tlwh, score, feat=None, feat_history=50, activation_wait=30, feature_warmup=5):

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
        self.feature_warmup = 0 if self.activation_wait == 0 else max(0, min(feature_warmup, self.activation_wait))
        self.pending = False
        self.pending_start_frame = None

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

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

    def activate(self, kalman_filter, frame_id, assign_id=True, feat=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        if assign_id:
            self.track_id = self.next_id()
            self.pending = False
            self.pending_start_frame = None
        else:
            self.track_id = self.next_pending_id()
            self.pending = True
            self.pending_start_frame = frame_id

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = assign_id
        self.frame_id = frame_id
        self.start_frame = frame_id

        if feat is not None and (assign_id or self.should_collect_features(frame_id)):
            self.update_features(feat)

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.pending = False
        self.pending_start_frame = None
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
            if not self.pending or self.should_collect_features(frame_id):
                self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = not self.pending

        self.score = new_track.score

    def should_collect_features(self, frame_id):
        if not self.pending or self.activation_wait <= 0:
            return False
        frames_elapsed = frame_id - self.start_frame
        warmup_start = max(0, self.activation_wait - self.feature_warmup)
        return frames_elapsed >= warmup_start

    def ready_for_promotion(self, frame_id):
        if not self.pending:
            return False
        frames_elapsed = frame_id - self.start_frame
        return frames_elapsed >= self.activation_wait

    def promote(self, reuse_id=None):
        if reuse_id is None:
            self.track_id = self.next_id()
        else:
            self.track_id = reuse_id
        self.pending = False
        self.pending_start_frame = None
        self.is_activated = True

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


class BoTSORT(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()
        STrack.reset_pending_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        lost_track_seconds = getattr(args, "lost_track_buffer_seconds", None)
        if lost_track_seconds is None:
            lost_track_seconds = self.buffer_size / max(frame_rate, 1e-6)
        self.max_time_lost = int(frame_rate * lost_track_seconds)
        self.activation_wait = max(0, getattr(args, "activation_wait", 0))
        feature_warmup = max(0, getattr(args, "activation_feature_warmup", 0))
        self.feature_warmup = min(feature_warmup, self.activation_wait) if self.activation_wait > 0 else 0
        self.overlap_iou_thresh = getattr(args, "overlap_iou_thresh", 0.5)
        self.returning_track_match_thresh = getattr(args, "returning_track_match_thresh", args.appearance_thresh)

        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)
        else:
            self.encoder = None
        self.reid_usage_count = 0
        self.reid_comparison_count = 0

        logger.info(
            "Tracker configuration -> activation_wait: {}, feature_warmup: {}, lost_track_seconds: {}, overlap_iou_thresh: {}".
            format(self.activation_wait, self.feature_warmup, lost_track_seconds, self.overlap_iou_thresh)
        )

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

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

        features_keep = None
        pending_needs_features = False
        if self.args.with_reid and len(dets) > 0:
            pending_needs_features = self._pending_tracks_need_features()

        detection_kwargs = dict(activation_wait=self.activation_wait, feature_warmup=self.feature_warmup)
        if len(dets) > 0:
            '''Detections'''
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, **detection_kwargs)
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

        # Determine whether we need ReID features for this frame
        need_reid_for_main = False
        if self.args.with_reid and len(detections) > 0 and len(tracked_stracks) > 0:
            tracked_iou_cost = matching.iou_distance(tracked_stracks, detections)
            need_reid_for_main = self._has_overlap(tracked_iou_cost)

        need_features = (
            self.args.with_reid
            and len(dets) > 0
            and (need_reid_for_main or pending_needs_features)
        )
        if need_features and self.encoder is not None:
            features_keep = self.encoder.inference(img, dets)
            for det, feat in zip(detections, features_keep):
                det.update_features(feat)
            self.reid_usage_count += 1
            reasons = []
            if need_reid_for_main:
                reasons.append("overlap")
            if pending_needs_features:
                reasons.append("pending_warmup")
            logger.info(
                "ReID inference triggered for {} (total usages: {})".format(
                    "+".join(reasons) if reasons else "unspecified", self.reid_usage_count
                )
            )

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid and need_reid_for_main and len(detections) > 0:
            if features_keep is None and self.encoder is not None:
                features_keep = self.encoder.inference(img, dets)
                for det, feat in zip(detections, features_keep):
                    det.update_features(feat)
                self.reid_usage_count += 1
                logger.info(
                    "ReID inference triggered for overlap (total usages: {})".format(self.reid_usage_count)
                )
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

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
                STrack(STrack.tlbr_to_tlwh(tlbr), s, **detection_kwargs)
                for (tlbr, s) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

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

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id, assign_id=False, feat=track.curr_feat)
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
        self._finalize_pending_tracks()
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks

    def _pending_tracks_need_features(self):
        for track in self.tracked_stracks:
            if getattr(track, "pending", False) and track.should_collect_features(self.frame_id):
                return True
        return False

    def _has_overlap(self, iou_costs):
        if iou_costs.size == 0:
            return False
        overlaps = 1 - iou_costs
        if overlaps.size == 0:
            return False
        return (
            np.any((overlaps > self.overlap_iou_thresh).sum(axis=0) > 1)
            or np.any((overlaps > self.overlap_iou_thresh).sum(axis=1) > 1)
        )

    @staticmethod
    def _cosine_distance(feat_a, feat_b):
        if feat_a is None or feat_b is None:
            return 1.0
        return 1.0 - float(np.dot(feat_a, feat_b))

    def _match_pending_with_lost(self, pending_track):
        if pending_track.smooth_feat is None:
            logger.info("Pending track lacks appearance features for comparison.")
            return None

        best_track = None
        best_dist = float("inf")
        comparisons = 0
        for lost in self.lost_stracks:
            if lost.smooth_feat is None:
                continue
            dist = self._cosine_distance(pending_track.smooth_feat, lost.smooth_feat)
            comparisons += 1
            if dist < best_dist:
                best_dist = dist
                best_track = lost

        if comparisons > 0:
            self.reid_comparison_count += comparisons
            logger.info(
                "Compared pending track with {} lost tracks (total comparisons: {})".format(
                    comparisons, self.reid_comparison_count
                )
            )
        else:
            logger.info("No eligible lost tracks available for appearance comparison.")

        if best_track is not None and best_dist <= self.returning_track_match_thresh:
            logger.info(
                "Reactivated lost track {} for returning object with appearance distance {:.4f}".format(
                    best_track.track_id, best_dist
                )
            )
            return best_track
        return None

    def _finalize_pending_tracks(self):
        if not self.tracked_stracks:
            return

        replaced_tracks = []
        tracks_to_add = []
        for track in list(self.tracked_stracks):
            if getattr(track, "pending", False) and track.ready_for_promotion(self.frame_id):
                matched = self._match_pending_with_lost(track) if self.args.with_reid else None
                if matched is not None:
                    matched.re_activate(track, self.frame_id, new_id=False)
                    matched.promote(reuse_id=matched.track_id)
                    matched.score = track.score
                    tracks_to_add.append(matched)
                    replaced_tracks.append(track)
                    if matched in self.lost_stracks:
                        self.lost_stracks.remove(matched)
                else:
                    track.promote()
                    logger.info(
                        "Assigned new track id {} after waiting {} frames".format(
                            track.track_id, self.activation_wait
                        )
                    )

        if replaced_tracks:
            for pending in replaced_tracks:
                pending.mark_removed()
            self.removed_stracks.extend(replaced_tracks)
            self.tracked_stracks = [t for t in self.tracked_stracks if t not in replaced_tracks]
            self.tracked_stracks.extend(tracks_to_add)

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
