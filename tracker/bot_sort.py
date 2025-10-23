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

    def __init__(self, tlwh, score, feat=None, feat_history=50,
                 activation_wait=30, appearance_warmup=5):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.public_track_id = -1
        self.waited_frames = 0
        self.activation_wait = max(0, int(activation_wait))
        self.appearance_warmup = max(0, int(min(appearance_warmup, self.activation_wait)))

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def needs_feature_collection(self):
        if self.public_track_id != -1:
            return False
        remaining = self.activation_wait - self.waited_frames
        return remaining <= self.appearance_warmup and self.activation_wait > 0

    def ready_for_public_id(self):
        return self.public_track_id == -1 and self.waited_frames >= self.activation_wait

    def set_public_track_id(self, track_id):
        self.public_track_id = track_id

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
        self.waited_frames = 0
        self.public_track_id = -1
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
        if self.public_track_id != -1:
            self.waited_frames = self.activation_wait
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
        if self.public_track_id == -1:
            self.waited_frames += 1

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
        self.frame_rate = frame_rate

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.activation_wait = getattr(args, "activation_wait_frames", 30)
        self.appearance_warmup = getattr(args, "activation_warmup_frames", 5)
        self.overlap_iou_thresh = getattr(args, "overlap_iou_thresh", 0.5)
        lost_time = getattr(args, "lost_track_time", 300.0)
        self.buffer_size = max(
            int(frame_rate * lost_time),
            int(frame_rate / 30.0 * args.track_buffer)
        )
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        self.reid_usage_counter = 0
        self.reid_comparisons = 0
        self.encoder = None

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)
            logger.info(
                "ReID enabled: activation_wait=%d, warmup=%d, lost_track_time=%.1fs",
                self.activation_wait,
                self.appearance_warmup,
                lost_time,
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

        if len(dets) > 0:
            '''Detections'''
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    s,
                    feat=None,
                    activation_wait=self.activation_wait,
                    appearance_warmup=self.appearance_warmup,
                )
                for tlbr, s in zip(dets, scores_keep)
            ]
        else:
            detections = []

        def extract_features_for_boxes(boxes, reason):
            if not self.args.with_reid or self.encoder is None:
                return []
            if len(boxes) == 0:
                return []
            feats = self.encoder.inference(img, boxes)
            self.reid_usage_counter += 1
            logger.info(
                "ReID inference #%d used for %d detections (%s)",
                self.reid_usage_counter,
                len(boxes),
                reason,
            )
            return feats

        def ensure_detection_features(indices, reason):
            if not self.args.with_reid or self.encoder is None:
                return
            valid_indices = [idx for idx in indices if 0 <= idx < len(detections)]
            missing = [idx for idx in valid_indices if detections[idx].curr_feat is None]
            if not missing:
                return
            det_boxes = [dets[idx] for idx in missing]
            feats = extract_features_for_boxes(det_boxes, reason)
            for idx, feat in zip(missing, feats):
                detections[idx].update_features(feat)

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
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        need_reid_for_assignment = False
        if self.args.with_reid and len(strack_pool) > 0 and len(detections) > 0:
            iou_overlap = 1.0 - ious_dists
            overlap_matrix = iou_overlap >= self.overlap_iou_thresh
            if np.any(overlap_matrix.sum(axis=0) > 1) or np.any(overlap_matrix.sum(axis=1) > 1):
                need_reid_for_assignment = True
                logger.debug(
                    "Potential ID switch detected (frame %d). Triggering ReID for association.",
                    self.frame_id,
                )

        if need_reid_for_assignment:
            ensure_detection_features(range(len(detections)), "crowd/overlap resolution")
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
            if track.needs_feature_collection():
                ensure_detection_features([idet], f"warmup for track {track.track_id}")
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if u_detection and self.args.with_reid:
            ensure_detection_features(u_detection, "new track introduction")

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

        need_reid_second = False
        if self.args.with_reid and len(r_tracked_stracks) > 0 and len(detections_second) > 0:
            iou_overlap_second = 1.0 - dists
            overlap_second = iou_overlap_second >= self.overlap_iou_thresh
            if np.any(overlap_second.sum(axis=0) > 1) or np.any(overlap_second.sum(axis=1) > 1):
                need_reid_second = True
                logger.debug(
                    "Potential ID switch detected in low-score association (frame %d).", self.frame_id
                )

        if need_reid_second:
            boxes = [STrack.tlwh_to_tlbr(det.tlwh) for det in detections_second]
            feats = extract_features_for_boxes(boxes, "low-score overlap resolution")
            for det, feat in zip(detections_second, feats):
                det.update_features(feat)
            emb_dists = matching.embedding_distance(r_tracked_stracks, detections_second) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            dists = np.minimum(dists, emb_dists)

        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.needs_feature_collection() and det.curr_feat is None:
                boxes = [STrack.tlwh_to_tlbr(det.tlwh)]
                feats = extract_features_for_boxes(boxes, f"warmup for track {track.track_id} (low score)")
                for feat in feats:
                    det.update_features(feat)
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

        need_reid_unconfirmed = False
        if self.args.with_reid and len(unconfirmed) > 0 and len(detections) > 0:
            iou_overlap_unconfirmed = 1.0 - ious_dists
            overlap_unconfirmed = iou_overlap_unconfirmed >= self.overlap_iou_thresh
            if np.any(overlap_unconfirmed.sum(axis=0) > 1) or np.any(overlap_unconfirmed.sum(axis=1) > 1):
                need_reid_unconfirmed = True
                logger.debug(
                    "Potential ID switch detected for unconfirmed tracks (frame %d).", self.frame_id
                )

        if need_reid_unconfirmed:
            ensure_detection_features(range(len(detections)), "unconfirmed overlap resolution")
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            if track.needs_feature_collection():
                ensure_detection_features([idet], f"warmup for track {track.track_id} (unconfirmed)")
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

        self._assign_public_ids()

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks

    def _assign_public_ids(self):
        for track in self.tracked_stracks:
            if track.ready_for_public_id():
                reused = False
                if self.args.with_reid and track.smooth_feat is not None:
                    reused = self._match_lost_track(track)
                if not reused:
                    track.set_public_track_id(track.track_id)
                    logger.info(
                        "Track %d assigned new public id %d after waiting %d frames.",
                        track.track_id,
                        track.public_track_id,
                        track.waited_frames,
                    )

    def _match_lost_track(self, track):
        candidates = [t for t in self.lost_stracks if t.public_track_id != -1 and t.smooth_feat is not None]
        if not candidates:
            logger.debug(
                "No lost track candidates available for track %d before ID assignment.",
                track.track_id,
            )
            return False

        logger.info(
            "Comparing track %d against %d lost tracks before assigning ID.",
            track.track_id,
            len(candidates),
        )
        self.reid_comparisons += len(candidates)

        dists = matching.embedding_distance(candidates, [track])
        if dists.size == 0:
            return False
        best_idx = int(np.argmin(dists[:, 0]))
        best_dist = float(dists[best_idx, 0])
        if best_dist < self.appearance_thresh:
            matched = candidates[best_idx]
            track.set_public_track_id(matched.public_track_id)
            logger.info(
                "Track %d reused public id %d (distance %.3f).",
                track.track_id,
                track.public_track_id,
                best_dist,
            )
            matched.mark_removed()
            self.removed_stracks.append(matched)
            self.lost_stracks = [t for t in self.lost_stracks if t.track_id != matched.track_id]
            return True

        logger.info(
            "Track %d did not match lost tracks (best distance %.3f).",
            track.track_id,
            best_dist,
        )
        return False


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
