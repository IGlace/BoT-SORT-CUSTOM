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

    def __init__(self, tlwh, score, feat=None, feat_history=50):

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

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])
        self.reid_ambiguity_thresh = getattr(args, 'reid_ambiguity_thresh', 0.05)
        self.reid_overlap_thresh = getattr(args, 'reid_overlap_thresh', 0.7)
        self.reid_min_track_age = getattr(args, 'reid_min_track_age', 4)
        self.total_reid_frames = 0
        self.total_reid_calls = 0
        self._reid_used_this_frame = False
        self._reid_calls_current_frame = 0

    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        pending_init_features = {}
        self._reid_used_this_frame = False
        self._reid_calls_current_frame = 0

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
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
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
        ious_dists = matching.iou_distance(strack_pool, detections)
        raw_ious_dists = ious_dists.copy()
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if self.args.with_reid:
            reid_det_indices, reid_track_indices = self._select_reid_candidates(raw_ious_dists, strack_pool, detections)
            if reid_det_indices:
                self._extract_reid_features(
                    img,
                    detections,
                    reid_det_indices,
                    self.frame_id,
                    stage="high-score",
                    raw_boxes=dets,
                    track_indices=reid_track_indices,
                    tracks=strack_pool
                )

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = self._compute_partial_embedding_distance(strack_pool, detections) / 2.0
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
            if self.args.with_reid and det.curr_feat is None and self._track_ready_for_init(track):
                pending_init_features[track.track_id] = (track, np.array(det.tlbr, dtype=np.float32))

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
            if self.args.with_reid and getattr(det, 'curr_feat', None) is None and self._track_ready_for_init(track):
                pending_init_features[track.track_id] = (track, np.array(det.tlbr, dtype=np.float32))

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        raw_ious_unconfirmed = ious_dists.copy()
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            reid_det_indices, reid_track_indices = self._select_reid_candidates(raw_ious_unconfirmed, unconfirmed, detections)
            if reid_det_indices:
                self._extract_reid_features(
                    img,
                    detections,
                    reid_det_indices,
                    self.frame_id,
                    stage="unconfirmed",
                    track_indices=reid_track_indices,
                    tracks=unconfirmed
                )
            emb_dists = self._compute_partial_embedding_distance(unconfirmed, detections) / 2.0
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
            if self.args.with_reid and detections[idet].curr_feat is None and self._track_ready_for_init(unconfirmed[itracked]):
                pending_init_features[unconfirmed[itracked].track_id] = (unconfirmed[itracked], np.array(detections[idet].tlbr, dtype=np.float32))
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

        if self.args.with_reid:
            self._process_pending_initial_features(img, pending_init_features, self.frame_id)

        if self._reid_used_this_frame:
            self.total_reid_frames += 1

        logger.info(
            "Frame {} summary: ReID used this frame: {} (calls this frame: {}, total frames with ReID: {}, total ReID calls: {})",
            self.frame_id,
            self._reid_used_this_frame,
            self._reid_calls_current_frame,
            self.total_reid_frames,
            self.total_reid_calls
        )
        logger.info("================================================")

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks

    def _select_reid_candidates(self, cost_matrix, tracks, detections):
        if cost_matrix.size == 0 or len(tracks) == 0 or len(detections) == 0:
            return [], []

        # Log the cost matrix
        logger.debug(
            "\nReID Selection Cost Matrix Details:\n"
            "Cost Matrix Shape: {}\n"
            "Cost Matrix:\n{}\n"
            "Track Count: {}\n"
            "Detection Count: {}\n",
            cost_matrix.shape,
            np.array2string(cost_matrix, precision=3, suppress_small=True),
            len(tracks),
            len(detections)
        )

        ambiguous_tracks = set()
        ambiguous_dets = set()
        cost_matrix = np.asarray(cost_matrix)

        # Track-driven ambiguity (multiple close detections)
        for track_idx, row in enumerate(cost_matrix):
            finite_mask = np.isfinite(row)
            valid_row = row[finite_mask]
            if valid_row.size < 2:
                continue
            order = np.argsort(valid_row)
            best = valid_row[order[0]]
            second = valid_row[order[1]]
            if second - best < self.reid_ambiguity_thresh:
                ambiguous_tracks.add(track_idx)
                candidate_indices = np.where(finite_mask)[0][order[:2]]
                ambiguous_dets.update(candidate_indices.tolist())

        # Detection-driven ambiguity and high overlap
        for det_idx, col in enumerate(cost_matrix.T):
            finite_mask = np.isfinite(col)
            valid_col = col[finite_mask]
            if valid_col.size >= 2:
                order = np.argsort(valid_col)
                best = valid_col[order[0]]
                second = valid_col[order[1]]
                if second - best < self.reid_ambiguity_thresh:
                    ambiguous_dets.add(det_idx)
                    candidate_tracks = np.where(finite_mask)[0][order[:2]]
                    ambiguous_tracks.update(candidate_tracks.tolist())

            iou_vals = 1.0 - col
            close_tracks = np.where(iou_vals > self.reid_overlap_thresh)[0]
            if close_tracks.size >= 2:
                ambiguous_dets.add(det_idx)
                ambiguous_tracks.update(close_tracks.tolist())

        return sorted(ambiguous_dets), sorted(ambiguous_tracks)

    def _extract_reid_features(self, img, detections, det_indices, frame_id, stage, raw_boxes=None, track_indices=None, tracks=None):
        if not det_indices:
            return
        if img is None or self.encoder is None:
            return
            
        # Debug logging for ReID selection
        logger.debug(
            "\nReID Selection Details (Frame {}, {} stage):\n" 
            "Selected Detection Indices: {}\n"
            "Selected Track Indices: {}\n"
            "Number of Total Detections: {}\n"
            "Number of Total Tracks: {}\n"
            "Detection Boxes: {}\n"
            "Track States: {}\n",
            frame_id,
            stage,
            det_indices,
            track_indices if track_indices is not None else "N/A",
            len(detections),
            len(tracks) if tracks is not None else "N/A",
            [det.tlwh.tolist() for det in detections] if detections else "N/A",
            [(t.track_id, t.state) for t in tracks] if tracks else "N/A"
        )

        unique_indices = sorted(set(det_indices))
        if raw_boxes is not None:
            boxes = np.asarray([raw_boxes[i] for i in unique_indices], dtype=np.float32)
        else:
            boxes = np.asarray([detections[i].tlbr for i in unique_indices], dtype=np.float32)
        if boxes.size == 0:
            return

        associated_tracks = []
        if track_indices is not None and tracks is not None:
            associated_tracks = [tracks[idx].track_id for idx in track_indices if idx < len(tracks)]

        self._reid_used_this_frame = True
        self._reid_calls_current_frame += len(unique_indices)
        self.total_reid_calls += len(unique_indices)

        logger.info(
            "Frame {}: ReID triggered at '{}' stage for detections {} (boxes {}). Tracks involved: {}",
            frame_id,
            stage,
            unique_indices,
            boxes.tolist(),
            associated_tracks if associated_tracks else "N/A"
        )

        feats = self.encoder.inference(img, boxes)
        for idx, feat in zip(unique_indices, feats):
            detections[idx].update_features(feat)

    @staticmethod
    def _compute_partial_embedding_distance(tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        track_indices = [idx for idx, track in enumerate(tracks) if track.smooth_feat is not None]
        det_indices = [idx for idx, det in enumerate(detections) if det.curr_feat is not None]

        emb_dists = np.ones((len(tracks), len(detections)), dtype=np.float32)
        if not track_indices or not det_indices:
            return emb_dists

        track_subset = [tracks[idx] for idx in track_indices]
        det_subset = [detections[idx] for idx in det_indices]
        sub_cost = matching.embedding_distance(track_subset, det_subset)

        for row_idx, track_idx in enumerate(track_indices):
            for col_idx, det_idx in enumerate(det_indices):
                emb_dists[track_idx, det_idx] = sub_cost[row_idx, col_idx]
        return emb_dists

    def _track_ready_for_init(self, track):
        track_age = (track.frame_id - track.start_frame + 1) if track.start_frame is not None else 0
        return track_age >= self.reid_min_track_age and track.smooth_feat is None

    def _process_pending_initial_features(self, img, pending_features, frame_id):
        if not pending_features or self.encoder is None:
            return
        boxes = np.asarray([item[1] for item in pending_features.values()], dtype=np.float32)
        if boxes.size == 0:
            return
        self._reid_used_this_frame = True
        self._reid_calls_current_frame += len(pending_features)
        self.total_reid_calls += len(pending_features)
        track_ids = [item[0].track_id for item in pending_features.values()]
        logger.info(
            "Frame {}: Capturing initial ReID features for confirmed tracks {}",
            frame_id,
            track_ids
        )
        feats = self.encoder.inference(img, boxes)
        for (track, _), feat in zip(pending_features.values(), feats):
            track.update_features(feat)


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
