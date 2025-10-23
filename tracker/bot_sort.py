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

    def __init__(
        self,
        tlwh,
        score,
        feat=None,
        feat_history=50,
        activation_wait=30,
        pre_activation_collect=5,
        is_detection_proxy=False,
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
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        self.activation_wait = activation_wait
        self.pre_activation_collect = pre_activation_collect
        self.frames_since_first_seen = 0
        self.public_track_id = -1
        self.pre_activation_features = deque([], maxlen=max(1, pre_activation_collect))
        self.is_detection_proxy = is_detection_proxy

        if feat is not None:
            self.update_features(feat, force=True)

    def update_features(self, feat, force=False):
        if feat is None:
            return
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat

        if self.is_detection_proxy:
            self.smooth_feat = feat
            return

        if self.public_track_id == -1 and not force:
            if self.frames_since_first_seen >= max(0, self.activation_wait - self.pre_activation_collect):
                self.pre_activation_features.append(feat)
            return

        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def commit_pre_activation_features(self):
        while self.pre_activation_features:
            feat = self.pre_activation_features.popleft()
            self.update_features(feat, force=True)

    def get_public_id(self):
        return self.public_track_id

    def ready_for_public_id(self):
        return self.public_track_id == -1 and self.frames_since_first_seen >= self.activation_wait

    def average_pre_activation_feature(self):
        if not self.pre_activation_features:
            return None
        feat = np.mean(np.stack(self.pre_activation_features, axis=0), axis=0)
        norm = np.linalg.norm(feat)
        if norm == 0:
            return None
        return feat / norm

    def copy_features_from(self, other_track):
        if other_track.smooth_feat is not None:
            self.smooth_feat = other_track.smooth_feat.copy()
        if other_track.curr_feat is not None:
            self.curr_feat = other_track.curr_feat.copy()
        if hasattr(other_track, "features"):
            self.features = deque(list(other_track.features), maxlen=self.features.maxlen)

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
        self.frames_since_first_seen = 1
        self.public_track_id = -1
        self.pre_activation_features.clear()

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, force=True)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
            self.public_track_id = self.track_id
        else:
            if self.public_track_id == -1:
                self.public_track_id = self.track_id
        self.frames_since_first_seen = max(self.frames_since_first_seen, self.activation_wait)
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
        self.frames_since_first_seen += 1

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


class BoTSORT(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.activation_wait = getattr(args, "activation_wait", 30)
        self.pre_activation_collect = getattr(args, "pre_activation_frames", 5)
        lost_seconds = getattr(args, "lost_track_buffer_seconds", 300.0)
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = max(self.buffer_size, int(frame_rate * lost_seconds))
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.overlap_iou_reid_thresh = getattr(args, "overlap_reid_iou_thresh", 0.6)

        self.reid_usage_count = 0
        self.reid_feature_total = 0

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

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

        if len(dets) > 0:
            '''Detections'''
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    s,
                    activation_wait=self.activation_wait,
                    pre_activation_collect=self.pre_activation_collect,
                    is_detection_proxy=True,
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

        # Associate with high score detection boxes using IoU first
        ious_dists = matching.iou_distance(strack_pool, detections)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        matches, u_track, u_detection = matching.linear_assignment(ious_dists, thresh=self.proximity_thresh)

        match_pairs = []
        track_detection_map = {}

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                match_pairs.append(("update", track, det))
            else:
                match_pairs.append(("reactivate", track, det))

        # ReID based association for unresolved matches
        reid_track_indices = list(u_track)
        reid_det_indices = list(u_detection)
        if self.args.with_reid and len(reid_det_indices) > 0:
            reid_track_candidates = [strack_pool[i] for i in reid_track_indices]
            reid_det_candidates = [detections[i] for i in reid_det_indices]

            self._extract_reid_features(img, reid_det_candidates, reason="unmatched detections")

            valid_track_mask = [t.smooth_feat is not None for t in reid_track_candidates]
            valid_tracks = [t for t, keep in zip(reid_track_candidates, valid_track_mask) if keep]
            if valid_tracks:
                emb_dists = matching.embedding_distance(valid_tracks, reid_det_candidates)
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                matches_reid, _, _ = matching.linear_assignment(emb_dists, thresh=self.appearance_thresh)

                matched_track_indices = set()
                matched_det_indices = set()
                valid_indices = [idx for idx, keep in enumerate(valid_track_mask) if keep]

                for local_track_idx, local_det_idx in matches_reid:
                    global_track_idx = valid_indices[local_track_idx]
                    track = reid_track_candidates[global_track_idx]
                    det = reid_det_candidates[local_det_idx]
                    if track.state == TrackState.Tracked:
                        match_pairs.append(("update", track, det))
                    else:
                        match_pairs.append(("reactivate", track, det))
                    matched_track_indices.add(global_track_idx)
                    matched_det_indices.add(local_det_idx)

                remaining_track_indices = [idx for idx in range(len(reid_track_candidates)) if idx not in matched_track_indices]
                remaining_det_indices = [idx for idx in range(len(reid_det_candidates)) if idx not in matched_det_indices]
                u_track = tuple(reid_track_indices[idx] for idx in remaining_track_indices)
                u_detection = tuple(reid_det_indices[idx] for idx in remaining_det_indices)
            else:
                u_track = tuple(reid_track_indices)
                u_detection = tuple(reid_det_indices)

        # Ensure appearance features for tracks close to activation
        self._ensure_pre_activation_features(img, match_pairs)

        for action, track, det in match_pairs:
            if action == "update":
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            track_detection_map[track.track_id] = det
            self._attempt_public_activation(track)

        self._handle_overlap_reid(img, track_detection_map)

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
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
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
            self._attempt_public_activation(track)

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

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            self._attempt_public_activation(track)

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

        if self.args.with_reid:
            logger.debug(
                f"Frame {self.frame_id}: cumulative ReID calls={self.reid_usage_count}, total candidates={self.reid_feature_total}."
            )

        return output_stracks

    def _ensure_pre_activation_features(self, img, match_pairs):
        if not self.args.with_reid:
            return

        detections_to_process = []
        seen = set()
        for _, track, det in match_pairs:
            if det.curr_feat is not None:
                continue
            if track.get_public_id() != -1:
                continue
            threshold_frame = max(0, track.activation_wait - track.pre_activation_collect)
            if track.frames_since_first_seen < threshold_frame:
                continue
            key = id(det)
            if key in seen:
                continue
            detections_to_process.append(det)
            seen.add(key)

        if detections_to_process:
            self._extract_reid_features(img, detections_to_process, reason="pre-activation feature collection")

    def _attempt_public_activation(self, track):
        if not track.ready_for_public_id():
            return

        candidate_feature = track.average_pre_activation_feature()
        logger.debug(
            f"Frame {self.frame_id}: track {track.track_id} ready for public id assignment; comparing with lost tracks."
        )

        matched_lost = None
        if self.args.with_reid and candidate_feature is not None and self.lost_stracks:
            available_lost = [lt for lt in self.lost_stracks if lt.smooth_feat is not None]
            if available_lost:
                original_feat = track.curr_feat
                track.curr_feat = candidate_feature
                dists = matching.embedding_distance(available_lost, [track])
                track.curr_feat = original_feat
                if dists.size > 0:
                    min_idx = int(np.argmin(dists[:, 0]))
                    min_dist = dists[min_idx, 0]
                    logger.debug(
                        f"Frame {self.frame_id}: best lost-track distance for track {track.track_id} is {min_dist:.4f}."
                    )
                    if min_dist <= self.appearance_thresh:
                        matched_lost = available_lost[min_idx]

        if matched_lost is not None:
            logger.info(
                f"Frame {self.frame_id}: reassigned track {track.track_id} to previous id {matched_lost.track_id} using lost-track appearance."
            )
            track.copy_features_from(matched_lost)
            track.commit_pre_activation_features()
            track.public_track_id = matched_lost.get_public_id() if matched_lost.get_public_id() != -1 else matched_lost.track_id
            track.track_id = matched_lost.track_id
            self._remove_lost_track(matched_lost)
        else:
            track.commit_pre_activation_features()
            track.public_track_id = track.track_id
            if candidate_feature is None:
                logger.warning(
                    f"Frame {self.frame_id}: track {track.track_id} activated without appearance feature; assigning new id."
                )
            else:
                logger.info(
                    f"Frame {self.frame_id}: assigned new public id {track.track_id} after waiting period."
                )

    def _handle_overlap_reid(self, img, track_detection_map):
        if not self.args.with_reid or len(track_detection_map) < 2:
            return

        tracks_with_dets = [track for track in self.tracked_stracks if track.track_id in track_detection_map]
        if len(tracks_with_dets) < 2:
            return

        overlap_candidates = {}
        for i in range(len(tracks_with_dets)):
            for j in range(i + 1, len(tracks_with_dets)):
                track_a = tracks_with_dets[i]
                track_b = tracks_with_dets[j]
                iou_dist = matching.iou_distance([track_a], [track_b])
                overlap = 1 - float(iou_dist[0][0])
                if overlap > self.overlap_iou_reid_thresh:
                    det_a = track_detection_map.get(track_a.track_id)
                    det_b = track_detection_map.get(track_b.track_id)
                    if det_a is not None and det_a.curr_feat is None:
                        overlap_candidates[id(det_a)] = (det_a, track_a)
                    if det_b is not None and det_b.curr_feat is None:
                        overlap_candidates[id(det_b)] = (det_b, track_b)

        if overlap_candidates:
            detections = [item[0] for item in overlap_candidates.values()]
            self._extract_reid_features(img, detections, reason="overlap potential", force=True)
            for det, parent_track in overlap_candidates.values():
                if det.curr_feat is not None:
                    parent_track.update_features(det.curr_feat, force=True)
                    self._attempt_public_activation(parent_track)

    def _extract_reid_features(self, img, track_candidates, reason, force=False):
        if not self.args.with_reid or not track_candidates:
            return

        boxes = np.asarray([STrack.tlwh_to_tlbr(track.tlwh) for track in track_candidates], dtype=np.float32)
        features = self.encoder.inference(img, boxes)
        self.reid_usage_count += 1
        self.reid_feature_total += len(track_candidates)
        logger.debug(
            f"Frame {self.frame_id}: ReID inference #{self.reid_usage_count} for {len(track_candidates)} candidates ({reason})."
        )
        for candidate, feat in zip(track_candidates, features):
            candidate.update_features(feat, force=force)

    def _remove_lost_track(self, lost_track):
        if lost_track in self.lost_stracks:
            self.lost_stracks.remove(lost_track)
        if lost_track in self.removed_stracks:
            self.removed_stracks.remove(lost_track)


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
