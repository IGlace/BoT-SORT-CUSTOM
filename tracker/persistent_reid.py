import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import threading
from loguru import logger


class PersistentReID:
    """
    Long-term person identity database for re-identification 
    after extended absences from the scene.
    """
    
    def __init__(self, max_age_minutes=30, min_track_confidence=0.8, max_identities=1000):
        self.max_age_minutes = max_age_minutes
        self.min_track_confidence = min_track_confidence
        self.max_identities = max_identities
        
        # Persistent storage: {track_id: identity_data}
        self.identity_db = {}
        self.lock = threading.RLock()
        
        # Identity data structure:
        # {
        #     'track_id': int,
        #     'features': list of np.ndarray,
        #     'last_seen': datetime,
        #     'first_seen': datetime,
        #     'total_observations': int,
        #     'confidence': float,
        #     'trajectory_summary': dict
        # }
    
    def add_identity(self, track_id, features, trajectory_info=None):
        """Add or update a person identity in the persistent database."""
        with self.lock:
            current_time = datetime.now()
            
            if track_id in self.identity_db:
                # Update existing identity
                identity = self.identity_db[track_id]
                identity['features'].extend(features)
                identity['last_seen'] = current_time
                identity['total_observations'] += len(features)
                identity['confidence'] = min(1.0, identity['confidence'] + 0.1)
                
                if trajectory_info:
                    identity['trajectory_summary'].update(trajectory_info)
                    
                # Limit feature history to prevent memory bloat
                if len(identity['features']) > 50:
                    identity['features'] = identity['features'][-50:]
                    
            else:
                # Add new identity
                if len(self.identity_db) >= self.max_identities:
                    self._cleanup_old_identities()
                
                self.identity_db[track_id] = {
                    'track_id': track_id,
                    'features': features,
                    'last_seen': current_time,
                    'first_seen': current_time,
                    'total_observations': len(features),
                    'confidence': self.min_track_confidence,
                    'trajectory_summary': trajectory_info or {}
                }
    
    def find_matching_identity(self, query_features, similarity_threshold=0.35, 
                             max_candidates=10, exclude_ids=None):
        """
        Find the best matching identity for given query features.
        
        Returns:
            list: [(track_id, similarity_score, identity_info)]
        """
        with self.lock:
            exclude_ids = exclude_ids or set()
            candidates = []
            
            # Clean up old identities first
            self._cleanup_old_identities()
            
            for track_id, identity in self.identity_db.items():
                if track_id in exclude_ids:
                    continue
                    
                if not identity['features']:
                    continue
                
                # Compute similarity with all stored features
                similarities = []
                for query_feat in query_features:
                    for stored_feat in identity['features'][-20:]:  # Use recent features
                        sim = self._cosine_similarity(query_feat, stored_feat)
                        if sim >= similarity_threshold:
                            similarities.append(sim)
                
                if similarities:
                    best_sim = max(similarities)
                    avg_sim = np.mean(similarities)
                    
                    # Weight by recentness and confidence
                    time_weight = self._compute_time_weight(identity['last_seen'])
                    final_score = avg_sim * identity['confidence'] * time_weight
                    
                    candidates.append((track_id, final_score, identity))
            
            # Sort by best similarity score
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:max_candidates]
    
    def _cosine_similarity(self, feat1, feat2):
        """Compute cosine similarity between two features."""
        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)
        return np.dot(feat1, feat2)
    
    def _compute_time_weight(self, last_seen):
        """Compute weight based on how recently the identity was seen."""
        hours_ago = (datetime.now() - last_seen).total_seconds() / 3600
        # Decay weight over time, but never below 0.1
        return max(0.1, np.exp(-hours_ago / 24))  # 24-hour decay constant
    
    def _cleanup_old_identities(self):
        """Remove identities that are too old or have low confidence."""
        current_time = datetime.now()
        to_remove = []
        
        for track_id, identity in self.identity_db.items():
            # Remove if too old
            if current_time - identity['last_seen'] > timedelta(minutes=self.max_age_minutes):
                to_remove.append(track_id)
            # Remove if low confidence and old
            elif (identity['confidence'] < self.min_track_confidence and 
                  current_time - identity['last_seen'] > timedelta(minutes=5)):
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.identity_db[track_id]
            logger.info(f"Removed old identity {track_id} from persistent database")
    
    def get_identity_info(self, track_id):
        """Get information about a specific identity."""
        with self.lock:
            return self.identity_db.get(track_id, None)
    
    def get_statistics(self):
        """Get database statistics for monitoring."""
        with self.lock:
            total_identities = len(self.identity_db)
            total_features = sum(len(id['features']) for id in self.identity_db.values())
            avg_confidence = np.mean([id['confidence'] for id in self.identity_db.values()]) if total_identities > 0 else 0
            
            return {
                'total_identities': total_identities,
                'total_features': total_features,
                'average_confidence': avg_confidence,
                'max_capacity': self.max_identities
            }
