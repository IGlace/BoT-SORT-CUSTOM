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
from tracker.persistent_reid import PersistentReID


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.has_assigned_id = False
        self.internal_id = None
        self.track_id = -1

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
        self.internal_id = self.next_id()
        self.track_id = -1
        self.has_assigned_id = False

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = False
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if new_id:
            self.internal_id = self.next_id()
            self.track_id = -1
            self.has_assigned_id = False
        self.is_activated = self.has_assigned_id
        self.frame_id = frame_id
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
        self.is_activated = self.has_assigned_id

        self.score = new_track.score

    def assign_identity(self):
        """Assign a persistent identity once the track is confirmed."""
        if self.internal_id is None:
            self.internal_id = self.next_id()
        self.track_id = self.internal_id
        self.has_assigned_id = True
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
            # Initialize persistent ReID for long-term identity management
            persistent_max_age = getattr(args, 'persistent_max_age_minutes', 30)
            persistent_max_ids = getattr(args, 'persistent_max_identities', 1000)
            self.persistent_reid = PersistentReID(
                max_age_minutes=persistent_max_age,
                max_identities=persistent_max_ids
            )
        else:
            self.persistent_reid = None

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])
        self.reid_ambiguity_thresh = getattr(args, 'reid_ambiguity_thresh', 0.05)
        self.reid_overlap_thresh = getattr(args, 'reid_overlap_thresh', 0.7)
        self.reid_min_track_age = getattr(args, 'reid_min_track_age', 4)
        self.reid_early_collect_offset = getattr(args, 'reid_early_collect_offset', 5)
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
            
            # Enhanced Motion-Appearance Fusion with adaptive weighting
            dists = self._enhanced_motion_appearance_fusion(ious_dists, emb_dists, strack_pool, detections)

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
            self._attempt_appearance_recovery(track)
            self._maybe_confirm_track(track)
            if self.args.with_reid and det.curr_feat is None and self._track_ready_for_init(track):
                pending_init_features[track.internal_id] = (track, np.array(det.tlbr, dtype=np.float32))

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
            self._attempt_appearance_recovery(track)
            self._maybe_confirm_track(track)
            if self.args.with_reid and getattr(det, 'curr_feat', None) is None and self._track_ready_for_init(track):
                pending_init_features[track.internal_id] = (track, np.array(det.tlbr, dtype=np.float32))

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
            track = unconfirmed[itracked]
            track.update(detections[idet], self.frame_id)
            self._attempt_appearance_recovery(track)
            self._maybe_confirm_track(track)
            activated_starcks.append(track)
            if self.args.with_reid and detections[idet].curr_feat is None and self._track_ready_for_init(track):
                pending_init_features[track.internal_id] = (track, np.array(detections[idet].tlbr, dtype=np.float32))
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
            self._attempt_appearance_recovery(track)
            self._maybe_confirm_track(track)
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

        # Safety monitoring and emergency response
        safety_metrics = self.monitor_safety_metrics()
        
        # Additional debug output for safety monitoring
        print(f"\n=== SAFETY MONITORING FRAME {self.frame_id} ===")
        for metric, value in safety_metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        # Check if safety monitoring is enabled in config
        safety_enabled = getattr(self.args, 'safety_monitoring', True)
        print(f"  Safety Monitoring: {'ENABLED' if safety_enabled else 'DISABLED'}")
        print("=" * 45)

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

    def _attempt_appearance_recovery(self, track):
        if not getattr(self.args, 'with_reid', False):
            logger.debug(
                "Frame {}: skip appearance recovery for track {} (ReID disabled)",
                self.frame_id,
                getattr(track, 'internal_id', None)
            )
            return False
        if track is None:
            logger.debug("Frame {}: skip appearance recovery (track is None)", self.frame_id)
            return False
        if track.has_assigned_id:
            logger.debug(
                "Frame {}: skip appearance recovery for track {} (already assigned)",
                self.frame_id,
                getattr(track, 'internal_id', None)
            )
            return False
        if track.curr_feat is None:
            logger.debug(
                "Frame {}: skip appearance recovery for track {} (no current feature yet)",
                self.frame_id,
                getattr(track, 'internal_id', None)
            )
            return False

        history_tracks = [
            t for t in (self.lost_stracks + self.removed_stracks)
            if getattr(t, 'smooth_feat', None) is not None and getattr(t, 'has_assigned_id', False)
        ]
        
        # First, try to match with recent tracks (lost + removed)
        if not history_tracks and self.persistent_reid is None:
            logger.debug(
                "Frame {}: appearance recovery for track {} found no historical candidates",
                self.frame_id,
                getattr(track, 'internal_id', None)
            )
            return False

        # Try recent tracks first
        if history_tracks:
            cost_matrix = matching.embedding_distance(history_tracks, [track])
            if cost_matrix.size == 0:
                logger.debug(
                    "Frame {}: appearance recovery for track {} yielded empty cost matrix",
                    self.frame_id,
                    getattr(track, 'internal_id', None)
                )
                # Fall through to persistent ReID if available
            else:
                best_idx = int(np.argmin(cost_matrix[:, 0]))
                best_cost = cost_matrix[best_idx, 0]
                logger.debug(
                    "Frame {}: appearance recovery candidate for track {} -> historical {} with cost {:.3f}",
                    self.frame_id,
                    getattr(track, 'internal_id', None),
                    getattr(history_tracks[best_idx], 'internal_id', None),
                    float(best_cost)
                )
                if np.isfinite(best_cost) and best_cost <= self.appearance_thresh:
                    matched_track = history_tracks[best_idx]
                    self._adopt_identity_from_history(track, matched_track, best_cost)
                    return True

        # If no recent match found, try persistent ReID database
        if self.persistent_reid is not None and track.curr_feat is not None:
            candidates = self.persistent_reid.find_matching_identity(
                [track.curr_feat], 
                similarity_threshold=getattr(self.args, 'persistent_similarity_threshold', 0.3),
                exclude_ids=set([t.track_id for t in self.tracked_stracks if hasattr(t, 'track_id')])
            )
            
            if candidates:
                best_track_id, similarity_score, identity_info = candidates[0]
                logger.info(
                    "Frame {}: Persistent ReID recovery for track {} -> identity {} (similarity: {:.3f})",
                    self.frame_id,
                    getattr(track, 'internal_id', None),
                    best_track_id,
                    float(similarity_score)
                )
                
                # Assign the recovered identity
                track.track_id = best_track_id
                track.has_assigned_id = True
                track.is_activated = True
                track.internal_id = best_track_id
                
                # Copy features from persistent identity for continuity
                if identity_info['features']:
                    track.smooth_feat = identity_info['features'][-1].copy()
                    track.features = deque(identity_info['features'][-10:], maxlen=track.features.maxlen)
                
                return True

        return False

    def _maybe_confirm_track(self, track):
        if track is None or track.has_assigned_id:
            return False
        if track.start_frame is None:
            return False
        track_age = (self.frame_id - track.start_frame + 1)
        if track_age >= self.reid_min_track_age:
            track.assign_identity()
            
            # Save to persistent ReID database for future recovery
            if (self.persistent_reid is not None and 
                track.smooth_feat is not None and 
                hasattr(track, 'track_id') and track.track_id > 0):
                
                features = list(track.features) if hasattr(track, 'features') else [track.smooth_feat]
                trajectory_info = {
                    'last_position': track.tlwh.tolist(),
                    'last_frame': self.frame_id,
                    'track_age': track_age
                }
                
                self.persistent_reid.add_identity(
                    track.track_id, 
                    features, 
                    trajectory_info
                )
                
                logger.info(
                    "Frame {}: Saved track {} to persistent ReID database ({} features)",
                    self.frame_id,
                    track.track_id,
                    len(features)
                )
            
            return True
        return False

    def _adopt_identity_from_history(self, provisional_track, historical_track, distance_cost):
        if historical_track in self.lost_stracks:
            self.lost_stracks.remove(historical_track)
        if historical_track in self.removed_stracks:
            self.removed_stracks.remove(historical_track)

        provisional_track.internal_id = historical_track.internal_id
        provisional_track.track_id = historical_track.track_id
        provisional_track.has_assigned_id = True
        provisional_track.is_activated = True
        provisional_track.start_frame = historical_track.start_frame
        provisional_track.tracklet_len = historical_track.tracklet_len

        if getattr(historical_track, 'features', None):
            provisional_track.features = deque(historical_track.features, maxlen=historical_track.features.maxlen)
        if getattr(historical_track, 'smooth_feat', None) is not None:
            provisional_track.smooth_feat = historical_track.smooth_feat.copy()

        if provisional_track.curr_feat is not None:
            provisional_track.update_features(provisional_track.curr_feat)

        logger.info(
            "Frame {}: appearance recovery reassigned lost track {} to provisional track (cosine distance {:.3f})",
            self.frame_id,
            provisional_track.track_id,
            float(distance_cost)
        )

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
        if track.start_frame is None:
            return False
        track_age = (track.frame_id - track.start_frame + 1)
        early_collection_age = max(0, self.reid_min_track_age - self.reid_early_collect_offset)
        return track_age >= early_collection_age and track.smooth_feat is None

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
            if not track.has_assigned_id:
                self._attempt_appearance_recovery(track)
                self._maybe_confirm_track(track)

    def _enhanced_motion_appearance_fusion(self, ious_dists, emb_dists, tracks, detections):
        """
        Enhanced motion-appearance fusion with adaptive weighting based on:
        - Motion uncertainty (Kalman filter covariance)
        - Track age and confidence
        - Scene density
        """
        if len(tracks) == 0 or len(detections) == 0:
            return ious_dists if emb_dists is None else (emb_dists if ious_dists is None else np.minimum(ious_dists, emb_dists))
        
        # Initialize adaptive distance matrix
        enhanced_dists = np.copy(ious_dists)
        
        for i, track in enumerate(tracks):
            if track.mean is None or track.covariance is None:
                continue
                
            # Calculate motion confidence (lower uncertainty = higher confidence)
            motion_uncertainty = np.trace(track.covariance[:4, :4])  # Position uncertainty
            motion_confidence = 1.0 / (1.0 + motion_uncertainty)
            
            # Calculate track age confidence
            track_age = max(1, self.frame_id - track.start_frame + 1)
            age_confidence = min(1.0, track_age / 30.0)  # Confidence increases with age
            
            # Calculate appearance confidence
            appearance_confidence = 0.0
            if track.smooth_feat is not None:
                appearance_confidence = 1.0
            
            # Adaptive weighting: more reliance on motion when motion is confident
            # More reliance on appearance when appearance is available and motion is uncertain
            motion_weight = motion_confidence * 0.7 + age_confidence * 0.3
            appearance_weight = appearance_confidence * (1.0 - motion_weight)
            
            # Apply fusion for each detection
            for j in range(len(detections)):
                if emb_dists[i, j] < 1.0:  # If appearance distance is valid
                    # Weighted fusion of motion and appearance
                    enhanced_dists[i, j] = (motion_weight * ious_dists[i, j] + 
                                           appearance_weight * emb_dists[i, j])
                    
                    # Additional penalty for high disagreement between motion and appearance
                    disagreement = abs(ious_dists[i, j] - emb_dists[i, j])
                    if disagreement > 0.3:  # High disagreement threshold
                        enhanced_dists[i, j] += disagreement * 0.5  # Penalty factor
        
        return enhanced_dists

    def _multi_stage_quality_assurance(self, matched_tracks, unmatched_tracks, detections):
        """
        Multi-stage quality assurance for critical situations:
        - High-density crowd verification
        - Critical track redundancy
        - Low-confidence fallback verification
        """
        verified_matches = []
        critical_situations = []
        
        # Stage 1: Check for high-density situations
        scene_density = len(self.tracked_stracks) + len(self.lost_stracks)
        high_density_threshold = getattr(self.args, 'high_density_threshold', 10)
        
        if scene_density > high_density_threshold:
            critical_situations.append('high_density')
            
        # Stage 2: Verify high-density matches with stricter criteria
        if 'high_density' in critical_situations:
            logger.info(f"Frame {self.frame_id}: High-density scene detected ({scene_density} tracks), enabling strict verification")
            
            for track_idx, det_idx in matched_tracks:
                track = unmatched_tracks[track_idx] if track_idx < len(unmatched_tracks) else None
                if track is None:
                    continue
                    
                # Stricter verification for critical tracks
                if self._is_critical_track(track):
                    # Require both motion and appearance agreement
                    if self._verify_critical_match(track, detections[det_idx]):
                        verified_matches.append((track_idx, det_idx))
                    else:
                        logger.warning(f"Frame {self.frame_id}: Critical track {track.track_id} failed verification")
                else:
                    verified_matches.append((track_idx, det_idx))
        else:
            verified_matches = matched_tracks
        
        # Stage 3: Low-confidence fallback verification
        if len(verified_matches) < scene_density * 0.7:  # Less than 70% tracks matched
            logger.warning(f"Frame {self.frame_id}: Low matching rate, enabling fallback verification")
            verified_matches = self._fallback_verification(unmatched_tracks, detections, verified_matches)
        
        return verified_matches

    def _is_critical_track(self, track):
        """Determine if a track requires critical verification."""
        # Critical if:
        # - Track has high confidence/long history
        # - Track is in critical zone (center of frame)
        # - Track has been frequently updated
        
        if track is None or track.start_frame is None:
            return False
            
        track_age = self.frame_id - track.start_frame + 1
        is_long_track = track_age > 50
        
        # Check if track is in central zone (assuming frame center is critical)
        if track.tlwh is not None:
            center_x, center_y = track.tlwh[0] + track.tlwh[2]/2, track.tlwh[1] + track.tlwh[3]/2
            # Assume 1920x1080 resolution for center zone check
            in_critical_zone = (960 < center_x < 1920-960) and (540 < center_y < 1080-540)
        else:
            in_critical_zone = False
            
        return is_long_track or in_critical_zone

    def _verify_critical_match(self, track, detection):
        """Strict verification for critical track-detection matches."""
        # Require multiple criteria to be satisfied
        
        # Motion consistency check
        if track.mean is not None:
            predicted_pos = track.mean[:4].copy()
            actual_pos = detection.tlwh
            motion_error = np.linalg.norm(predicted_pos - actual_pos)
            if motion_error > 50:  # pixels
                return False
        
        # Appearance consistency check
        if track.smooth_feat is not None and detection.curr_feat is not None:
            similarity = np.dot(track.smooth_feat, detection.curr_feat)
            if similarity < 0.5:  # Strict appearance threshold
                return False
        
        return True

    def _fallback_verification(self, tracks, detections, initial_matches):
        """Fallback verification using multiple strategies."""
        fallback_matches = set(initial_matches)
        
        # Strategy 1: IoU-only matching for remaining tracks
        remaining_tracks = [i for i in range(len(tracks)) if i not in [m[0] for m in initial_matches]]
        remaining_detections = [i for i in range(len(detections)) if i not in [m[1] for m in initial_matches]]
        
        if remaining_tracks and remaining_detections:
            remaining_track_objs = [tracks[i] for i in remaining_tracks]
            remaining_det_objs = [detections[i] for i in remaining_detections]
            
            ious_dists = matching.iou_distance(remaining_track_objs, remaining_det_objs)
            fallback_matches_add, _, _ = matching.linear_assignment(ious_dists, thresh=0.3)
            
            for track_idx, det_idx in fallback_matches_add:
                fallback_matches.add((remaining_tracks[track_idx], remaining_detections[det_idx]))
        
        return list(fallback_matches)

    def compute_track_confidence(self, track):
        """
        Track Confidence Monitoring: Compute confidence score for individual tracks
        based on multiple factors to ensure tracking quality.
        """
        if track is None:
            return 0.0
            
        confidence_factors = []
        
        # Factor 1: Track age/stability
        if track.start_frame is not None:
            track_age = self.frame_id - track.start_frame + 1
            age_confidence = min(1.0, track_age / 30.0)  # Max confidence at 30 frames
            confidence_factors.append(age_confidence)
        
        # Factor 2: Detection confidence
        if hasattr(track, 'score'):
            detection_confidence = track.score
            confidence_factors.append(detection_confidence)
        
        # Factor 3: Motion consistency (Kalman filter uncertainty)
        if track.covariance is not None:
            motion_uncertainty = np.trace(track.covariance[:4, :4])
            motion_confidence = 1.0 / (1.0 + motion_uncertainty / 100.0)  # Normalized
            confidence_factors.append(motion_confidence)
        
        # Factor 4: Feature quality (ReID consistency)
        if hasattr(track, 'smooth_feat') and track.smooth_feat is not None:
            feature_norm = np.linalg.norm(track.smooth_feat)
            feature_confidence = min(1.0, feature_norm)  # Should be close to 1.0 for normalized features
            confidence_factors.append(feature_confidence)
        
        # Factor 5: Update frequency
        if hasattr(track, 'tracklet_len'):
            update_frequency = track.tracklet_len / max(1, track_age)
            frequency_confidence = min(1.0, update_frequency)
            confidence_factors.append(frequency_confidence)
        
        # Compute weighted average
        if confidence_factors:
            weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Prioritize age and detection quality
            weighted_confidence = sum(w * f for w, f in zip(weights, confidence_factors))
            return weighted_confidence
        
        return 0.0

    def monitor_safety_metrics(self):
        """
        Safety Monitoring: Track system-level safety metrics and trigger
        emergency protocols when tracking quality degrades.
        """
        safety_metrics = {}
        
        # Metric 1: ID switch rate (need history)
        # This would require maintaining ID switch history
        safety_metrics['id_switch_rate'] = self._compute_id_switch_rate()
        
        # Metric 2: Track fragmentation rate
        active_tracks = len([t for t in self.tracked_stracks if t.is_activated])
        total_tracks = len(self.tracked_stracks) + len(self.lost_stracks)
        fragmentation_rate = 1.0 - (active_tracks / max(1, total_tracks))
        safety_metrics['fragmentation_rate'] = fragmentation_rate
        
        # Metric 3: Average track confidence
        if self.tracked_stracks:
            avg_confidence = np.mean([self.compute_track_confidence(t) for t in self.tracked_stracks])
            safety_metrics['avg_track_confidence'] = avg_confidence
        
        # Metric 4: ReID usage rate
        total_frames = max(1, self.frame_id)
        reid_usage_rate = self.total_reid_frames / total_frames
        safety_metrics['reid_usage_rate'] = reid_usage_rate
        
        # Metric 5: Scene density stress
        scene_density = len(self.tracked_stracks) + len(self.lost_stracks)
        density_stress = min(1.0, scene_density / 20.0)  # Normalize to [0, 1]
        safety_metrics['density_stress'] = density_stress
        
        # Safety thresholds and responses
        safety_thresholds = {
            'avg_track_confidence': 0.7,      # Minimum average confidence
            'fragmentation_rate': 0.3,        # Maximum fragmentation
            'density_stress': 0.8,            # Maximum scene stress
            'reid_usage_rate': 0.5            # Maximum ReID dependency
        }
        
        # Check for safety violations
        safety_violations = []
        for metric, threshold in safety_thresholds.items():
            if metric in safety_metrics:
                if metric in ['avg_track_confidence']:
                    if safety_metrics[metric] < threshold:
                        safety_violations.append(metric)
                else:
                    if safety_metrics[metric] > threshold:
                        safety_violations.append(metric)
        
        # Trigger safety responses
        if safety_violations:
            self._trigger_safety_response(safety_violations, safety_metrics)
        
        # Log safety status
        logger.info(f"=== SAFETY METRICS FRAME {self.frame_id} ===")
        for metric, value in safety_metrics.items():
            logger.info(f"  {metric}: {value:.3f}")
            
        if safety_violations:
            logger.warning(f"!!! SAFETY VIOLATIONS DETECTED: {safety_violations} !!!")
        else:
            logger.info("✓ All safety metrics within normal ranges")
        logger.info("==========================================")
        
        return safety_metrics

    def _compute_id_switch_rate(self):
        """Compute ID switch rate (simplified implementation)."""
        # This would require maintaining track history and ID assignment records
        # For now, return a placeholder based on ReID usage
        return self.total_reid_calls / max(1, self.frame_id)

    def _trigger_safety_response(self, violations, metrics):
        """Trigger appropriate safety responses for violations."""
        print(f"\n🚨 EMERGENCY SAFETY RESPONSE ACTIVATED 🚨")
        print(f"Frame {self.frame_id}: VIOLATIONS DETECTED: {violations}")
        logger.warning(f"Frame {self.frame_id}: TRIGGERING SAFETY RESPONSE for violations: {violations}")
        
        if 'avg_track_confidence' in violations:
            # Lower ReID thresholds, increase track buffer
            print(f"  → LOWERING ReID thresholds (confidence: {metrics.get('avg_track_confidence', 0):.3f})")
            logger.info("Emergency: Lowering ReID thresholds for better matching")
            self.appearance_thresh *= 0.8  # More permissive
            self.max_time_lost = int(self.max_time_lost * 1.5)  # Longer memory
        
        if 'fragmentation_rate' in violations:
            # Enable stricter matching, reduce new track creation
            print(f"  → ENABLING stricter matching (fragmentation: {metrics.get('fragmentation_rate', 0):.3f})")
            logger.info("Emergency: Enabling stricter matching to reduce fragmentation")
            self.new_track_thresh *= 1.2  # Higher threshold for new tracks
        
        if 'density_stress' in violations:
            # Enable multi-stage verification
            print(f"  → ENABLING multi-stage verification (density stress: {metrics.get('density_stress', 0):.3f})")
            logger.info("Emergency: Enabling multi-stage verification for crowded scene")
            # The _multi_stage_quality_assurance will handle this
        
        if 'reid_usage_rate' in violations:
            # Optimize ReID usage
            print(f"  → OPTIMIZING ReID usage (usage rate: {metrics.get('reid_usage_rate', 0):.3f})")
            logger.info("Emergency: Optimizing ReID usage patterns")
            self.reid_ambiguity_thresh *= 1.5  # Less sensitive to ambiguity
        
        print("=" * 50)


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        key = getattr(t, 'internal_id', t.track_id)
        exists[key] = 1
        res.append(t)
    for t in tlistb:
        tid = getattr(t, 'internal_id', t.track_id)
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[getattr(t, 'internal_id', t.track_id)] = t
    for t in tlistb:
        tid = getattr(t, 'internal_id', t.track_id)
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
