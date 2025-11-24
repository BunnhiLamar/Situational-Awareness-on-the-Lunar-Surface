import numpy as np
import cv2
class Helper:
    def __init__(self,args):
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
        self.K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        # baseline = float(lines[1])

        self.depth_scale = float(lines[1].strip())
        self.fx = self.K[0, 0]  # 1108.512
        self.fy = self.K[1, 1]  # 1108.512
        self.cx = self.K[0, 2]  # 640.0
        self.cy = self.K[1, 2]  # 360.0

    def pixel_to_3d(self,x, y, depth_value):
        """Convert pixel + depth to 3D point"""
        # z = depth_value * self.depth_scale
        z = depth_value
        x_3d = (x - self.cx) * z / self.fx
        y_3d = (y - self.cy) * z / self.fy
        return np.array([x_3d, y_3d, z])

    def distance_between_points(self,depth_map, p1, p2):
        """Calculate 3D distance between two pixels"""
        point1 = self.pixel_to_3d(p1[0], p1[1], depth_map[p1[1], p1[0]])
        point2 = self.pixel_to_3d(p2[0], p2[1], depth_map[p2[1], p2[0]])
        return np.linalg.norm(point2 - point1)

    def iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        box format: (x1, y1, x2, y2)
        """
        # Intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0



    def iou_batch(self,boxes1, boxes2):
        """
        Calculate IoU between all pairs of boxes
        boxes1: (N, 4) array
        boxes2: (M, 4) array
        Returns: (N, M) IoU matrix
        """
        # Expand dimensions for broadcasting
        boxes1 = boxes1[:, None, :]  # (N, 1, 4)
        boxes2 = boxes2[None, :, :]  # (1, M, 4)
        
        # Intersection
        x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Union
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - intersection
        
        return np.where(union > 0, intersection / union, 0)
    def non_max_suppression(self, boxes: list, 
                        scores: list, 
                        iou_threshold: float = 0.7) -> list:
        """
        Perform non-max suppression on a set of bounding boxes and corresponding scores.

        Args:
            boxes (list): A list of bounding boxes in the format [[xmin, ymin, xmax, ymax], ...].

            iou_threshold (float): The IoU (Intersection over Union) threshold for merging bounding boxes.

        Returns:
        list: A list of indices of the boxes to keep after non-max suppression.
        """
        scores = scores.cpu().numpy()
        print(boxes, scores)
        indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores= scores, score_threshold=0.3, nms_threshold=iou_threshold)
        return sorted(indices)
        

        # num_boxes = len(boxes)
        # selected_indices = []

        # for i in range(num_boxes):
        #     print(selected_indices)
        #     if i in selected_indices:
        #         continue

        #     selected_indices.append(i)
            
        #     for j in range(i + 1, num_boxes):
        #         iou_val = self.iou(boxes[i], boxes[j])

        #         if iou_val < iou_threshold:
        #             selected_indices.append(j)

        # return selected_indices
# distance = distance_between_points(depth_map, (x1,y1), (x2,y2))