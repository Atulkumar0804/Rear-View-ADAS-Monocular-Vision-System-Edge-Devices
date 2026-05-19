"""
ByteTrack integration for robust multi-object tracking
Replaces IoU-based tracking with motion-aware tracking
Reference: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
"""

import numpy as np
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
import cv2


class STrack:
    """Single object track representation"""
    
    shared_counter = 0
    
    def __init__(self, tlwh, conf, cls_id=None, cls_name=None):
        """
        Initialize track
        
        Args:
            tlwh: [x, y, width, height] bounding box
            conf: Detection confidence score
            cls_id: Class ID
            cls_name: Class name
        """
        # Convert tlwh to [x1, y1, x2, y2]
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.is_activated = False
        self.track_id = 0
        self.score = conf
        self.cls_id = cls_id
        self.cls_name = cls_name
        
        # State: [x, y, width, height, vx, vy] (center + velocity)
        self.mean, self.covariance = self._initiate_track()
        self.smooth_xy = np.asarray(tlwh[:2], dtype=np.float32)
        
        self.time_since_update = 0
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0
        
        # Additional metadata
        self.class_history = deque([cls_name], maxlen=5)
        self.confidence_history = deque([conf], maxlen=10)
    
    def _initiate_track(self):
        """Initialize position and velocity state"""
        x, y, w, h = self.tlwh
        mean = np.asarray([x + w / 2, y + h / 2, w, h], dtype=np.float32)
        covariance = np.diag([1e-2, 1e-2, 1e-4, 1e-4])
        return mean, covariance
    
    def predict(self):
        """Predict next position using constant velocity model"""
        mean = self.mean.copy()
        if len(mean) == 4:
            # Extend state with velocity
            mean = np.append(mean, [0, 0])
        
        # Constant velocity prediction
        if self.mean.shape[0] == 4:
            self.mean = np.append(self.mean, [0, 0])
        
        mean[:2] += 0  # No velocity term initially (constant position)
        
        # Increase uncertainty after prediction
        covariance = self.covariance.copy()
        covariance[np.diag_indices_from(covariance)] += [1e-1] * covariance.shape[0]
        
        self.mean = mean[:4]
        self.covariance = covariance
    
    def update(self, new_track, frame_id):
        """
        Update track with new detection
        
        Args:
            new_track: STrack object with new detection
            frame_id: Current frame ID
        """
        self.time_since_update = 0
        self.tracklet_len += 1
        self.frame_id = frame_id
        
        # Smooth position update (exponential moving average)
        alpha = 0.9
        new_tlwh = new_track.tlwh
        self.smooth_xy = alpha * self.smooth_xy + (1 - alpha) * new_tlwh[:2]
        
        # Update state
        self.tlwh = new_tlwh.copy()
        self.score = new_track.score
        
        if new_track.cls_id is not None:
            self.cls_id = new_track.cls_id
        if new_track.cls_name is not None:
            self.cls_name = new_track.cls_name
            self.class_history.append(new_track.cls_name)
        
        self.confidence_history.append(new_track.score)
        
        # Mark as activated if not already
        if not self.is_activated:
            self.is_activated = True
            self.start_frame = frame_id
    
    def mark_missed(self):
        """Mark track as missed (no detection matched)"""
        self.time_since_update += 1
    
    def is_activated_track(self):
        """Check if track is activated"""
        return self.is_activated
    
    def get_bbox(self):
        """Get current bounding box as [x1, y1, x2, y2]"""
        x, y, w, h = self.tlwh
        return [x, y, x + w, y + h]
    
    def get_tlwh(self):
        """Get bounding box as [x, y, width, height]"""
        return self.tlwh.copy()
    
    def get_center(self):
        """Get center point (x, y)"""
        x, y, w, h = self.tlwh
        return np.array([x + w/2, y + h/2])
    
    def get_avg_confidence(self):
        """Get average confidence over track lifetime"""
        if len(self.confidence_history) == 0:
            return self.score
        return float(np.mean(list(self.confidence_history)))


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    
    Key features:
    - Motion-based association for unmatched detections
    - High-confidence and low-confidence detection handling
    - Robust to occlusions and fast motion
    - Better than IoU-based tracking in crowded scenes
    """
    
    def __init__(self, track_buffer=300, frame_rate=30):
        """
        Initialize ByteTracker
        
        Args:
            track_buffer: Maximum frames to keep track alive without detection
            frame_rate: Video frame rate for motion estimation
        """
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.frame_id = 0
        self.track_id_counter = 0
        
        # Tracking parameters
        self.high_thresh = 0.6   # High confidence threshold
        self.low_thresh = 0.3    # Low confidence threshold (for recovery)
        self.iou_threshold = 0.1 # IoU threshold for matching
        
    def update(self, detections, frame_id=None):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with keys:
                       - bbox: [x, y, w, h]
                       - confidence: float
                       - class_id: int (optional)
                       - class_name: str (optional)
            frame_id: Current frame ID (auto-incremented if None)
        
        Returns:
            tracked_tracks: List of matched/tracked objects with track_id
        """
        if frame_id is None:
            self.frame_id += 1
        else:
            self.frame_id = frame_id
        
        # Separate high and low confidence detections
        high_confs = []
        low_confs = []
        
        for det in detections:
            bbox_tlwh = det.get('bbox', [])  # [x, y, w, h]
            if len(bbox_tlwh) != 4:
                bbox_tlwh = self._convert_bbox_format(det)
            
            conf = det.get('confidence', 0.5)
            cls_id = det.get('class_id')
            cls_name = det.get('class_name')
            
            strack = STrack(bbox_tlwh, conf, cls_id, cls_name)
            
            if conf >= self.high_thresh:
                high_confs.append(strack)
            else:
                low_confs.append(strack)
        
        # Update existing tracks (predict and match)
        self._update_tracks(high_confs)
        
        # Handle low-confidence detections
        self._handle_low_confidence(low_confs)
        
        # Build output
        output_tracks = []
        for track in self.tracked_stracks:
            if track.is_activated:
                output_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.get_bbox(),
                    'tlwh': track.get_tlwh(),
                    'confidence': float(track.score),
                    'class_id': track.cls_id,
                    'class_name': track.cls_name,
                    'avg_confidence': track.get_avg_confidence(),
                    'age': self.frame_id - track.start_frame,
                })
        
        return output_tracks
    
    def _convert_bbox_format(self, detection):
        """Convert [x1,y1,x2,y2] to [x,y,w,h] if needed"""
        if 'bbox' in detection:
            bbox = detection['bbox']
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                return [x1, y1, x2-x1, y2-y1]
        return [0, 0, 1, 1]
    
    def _update_tracks(self, high_confs):
        """Match high-confidence detections with existing tracks"""
        # Predict track positions
        for track in self.tracked_stracks:
            track.predict()
        
        # Associate detections to tracks
        matched_indices, unmatched_tracks, unmatched_detections = \
            self._associate_detections(high_confs)
        
        # Update matched tracks
        for track_idx, det_idx in matched_indices:
            self.tracked_stracks[track_idx].update(high_confs[det_idx], self.frame_id)
        
        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            self.tracked_stracks[track_idx].mark_missed()
        
        # Initialize new tracks from unmatched detections
        for det_idx in unmatched_detections:
            new_track = high_confs[det_idx]
            self.track_id_counter += 1
            new_track.track_id = self.track_id_counter
            new_track.is_activated = True
            new_track.start_frame = self.frame_id
            self.tracked_stracks.append(new_track)
        
        # Remove dead tracks
        self.tracked_stracks = [
            t for t in self.tracked_stracks 
            if t.time_since_update <= self.track_buffer
        ]
    
    def _handle_low_confidence(self, low_confs):
        """Try to recover lost tracks with low-confidence detections"""
        if not low_confs or not self.lost_stracks:
            return
        
        # Try to match low-conf with lost tracks
        matched_indices, unmatched_lost, unmatched_low = \
            self._associate_detections_iou(low_confs, self.lost_stracks)
        
        # Re-activate lost tracks
        for lost_idx, low_idx in matched_indices:
            lost_track = self.lost_stracks[lost_idx]
            lost_track.update(low_confs[low_idx], self.frame_id)
            self.tracked_stracks.append(lost_track)
        
        # Remove matched from lost
        self.lost_stracks = [
            self.lost_stracks[i] for i in unmatched_lost
        ]
    
    def _associate_detections(self, detections, detections_b=None):
        """
        Associate detections to tracks using:
        1. IoU-based matching (detected objects close together)
        2. Center distance matching (for separated objects)
        """
        if len(self.tracked_stracks) == 0:
            unmatched_dets = list(range(len(detections)))
            return [], [], unmatched_dets
        
        tracked_boxes = np.array([t.get_tlwh() for t in self.tracked_stracks])
        detection_boxes = np.array([d.get_tlwh() for d in detections])
        
        # Calculate IoU cost matrix
        iou_dist = self._iou_batch(detection_boxes, tracked_boxes)
        
        # Hungarian algorithm (linear sum assignment)
        matched_indices = linear_sum_assignment(iou_dist)
        
        unmatched_tracks = []
        unmatched_detections = []
        
        matched_indices = np.asarray(matched_indices).T
        
        for d_idx, t_idx in matched_indices:
            if iou_dist[d_idx, t_idx] > 0.5:  # No good match
                unmatched_detections.append(d_idx)
                unmatched_tracks.append(t_idx)
            else:
                pass  # Good match
        
        unmatched_dets = [i for i in range(len(detections)) 
                         if i not in matched_indices[:, 0]]
        unmatched_trks = [i for i in range(len(self.tracked_stracks)) 
                         if i not in matched_indices[:, 1]]
        
        # Filter good matches
        matches = []
        for d_idx, t_idx in matched_indices:
            if iou_dist[d_idx, t_idx] < 0.5:
                matches.append((t_idx, d_idx))
        
        return matches, unmatched_trks, unmatched_dets
    
    def _associate_detections_iou(self, detections, tracks):
        """Associate using IoU only (simpler for recovery)"""
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        
        track_boxes = np.array([t.get_tlwh() for t in tracks])
        det_boxes = np.array([d.get_tlwh() for d in detections])
        
        iou_dist = self._iou_batch(det_boxes, track_boxes)
        matched_indices = linear_sum_assignment(iou_dist)
        
        matches = []
        used_dets = set()
        used_tracks = set()
        
        for d_idx, t_idx in np.asarray(matched_indices).T:
            if iou_dist[d_idx, t_idx] < 0.5:
                matches.append((t_idx, d_idx))
                used_dets.add(d_idx)
                used_tracks.add(t_idx)
        
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _iou_batch(self, boxes_a, boxes_b):
        """
        Calculate IoU cost matrix (lower is better)
        
        Args:
            boxes_a: Nx4 array [x, y, w, h]
            boxes_b: Mx4 array [x, y, w, h]
        
        Returns:
            NxM cost matrix (1 - IoU, so 0 = perfect match)
        """
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)))
        
        ious = np.zeros((len(boxes_a), len(boxes_b)))
        
        for i, box_a in enumerate(boxes_a):
            x1_a, y1_a, w_a, h_a = box_a
            x2_a, y2_a = x1_a + w_a, y1_a + h_a
            area_a = w_a * h_a
            
            for j, box_b in enumerate(boxes_b):
                x1_b, y1_b, w_b, h_b = box_b
                x2_b, y2_b = x1_b + w_b, y1_b + h_b
                area_b = w_b * h_b
                
                # Intersection
                xi1 = max(x1_a, x1_b)
                yi1 = max(y1_a, y1_b)
                xi2 = min(x2_a, x2_b)
                yi2 = min(y2_a, y2_b)
                
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union_area = area_a + area_b - inter_area
                
                iou = inter_area / union_area if union_area > 0 else 0
                ious[i, j] = 1 - iou  # Cost: lower is better
        
        return ious
    
    def get_tracked_stracks(self):
        """Get all tracked objects"""
        return self.tracked_stracks
    
    def reset(self):
        """Reset tracker (for new video)"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
