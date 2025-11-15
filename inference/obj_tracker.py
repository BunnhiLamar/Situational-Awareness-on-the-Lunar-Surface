import numpy as np
from typing import Tuple, Optional, List
from kalman_filter import KalmanFilter
class MultiObjectTracker:
    def __init__(self, max_lost_frames: int=50,
                    iou_threshold: float=0.3):
        self.trackers ={}
        self.next_id = 0
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.lost_counts = {}

    def calculate_iou(self, bbox1: np.ndarray,bbox2: np.ndarray) -> float:
        """
        calculate iou between the 2 bounding boxes

        args:
             bbox1, bbox2: bounding boxes [xmin, ymin, xmax, y max]
        returns:
            iou
        """
        x1_min , y1_min, x1_max, y1_max = bbox1[:4]
        x2_min, y2_min, x2_max, y2_max = bbox2[:4]

        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)

        #calculate union
        area1 = (x1_max-x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[np.ndarray]) -> dict:
        """
        update trackers with new detections
        args:
            detections
        """
        # predict step for all trackers
        predictions = {}
        
        for track_id, tracker in self.trackers.items():
            predictions[track_id] = tracker.predict()

        #match detections to existing tracks
        matches = []
        unmatched_detection_indices = set(range(len(detections)))
        unmatched_track_ids = set(self.trackers.key())

        if len(detections) > 0 and len(self.trackers) > 0:
            # create iou matrix
            track_ids_list = list(self.trackers_keys())
            iou_matrix = np.zeros((len(detections), len(track_ids_list)))
            for i, detection in enumerate(detections):
                for j, track_id in enumerate(track_ids_list):
                    pred_bbox = predictions[track_id][:5, 0]
                    iou_matrix[i,j] = self.calculate_iou(detection, pred_bbox)

            # Greedy matching
            while iou_matrix.size > 0 :
                #find maximum iou
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break

                # get idx of max iou
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                det_idx_in_matrix = max_idx[0]
                track_idx_in_matrix = max_idx[1]

                # back to original idx
                remaining_det_indices = sorted(list(unmatched_detection_indices))
                remaining_track_indices = [tid for tid in track_ids_list if tid in unmatched_track_ids]

                det_idx = remaining_det_indices[det_idx_in_matrix]
                track_id = remaining_track_indices[track_idx_in_matrix]

                # record matches
                matches.append((det_idx, track_id))
                unmatched_detection_indices.remove(det_idx)
                unmatched_track_ids.remove(track_id)
                
                #rm matched row col 
                iou_matrix=np.delete(iou_matrix, det_idx, axis=0)
                iou_matrix = np.delete(iou_matrix, track_id, axis= 1)

        # update matched tracks
        for det_idx, track_id in matches:
            self.trackers[track_id].update(detections[det_idx])
            self.lost_counts[track_id] = 0
        
        #create new tracks for unmatched detections
        for det_idx in unmatched_detection_indices:
            new_id = self.next_id
            self.next_id +=1
            self.trackers[new_id] = KalmanFilter(detections[det_idx])
            self.lost_counts[new_id] = 0
         #handle unmatched tracks
        tracks_to_remove = []
        for track_id in unmatched_track_ids:
            self.lost_counts[track_id] +=1
            if self.lost_counts[track_id] > self.max_lost_frames:
                tracks_to_remove.append(track_id)
        
        # remove lost tracks
        for track_id in tracks_to_remove:
            del self.trackers[track_id]
            del self.lost_counts[track_id]

        # current states
        results = {}
        for track_id, tracker in self.trackers.items():
            results[track_id] = tracker.state.copy()
        return results
        
