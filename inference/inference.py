import os, sys, glob, argparse, re, imageio, cv2
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from FoundationStereo.scripts.run_demo import *
from natsort import natsorted
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from scipy.spatial.distance import pdist, squareform

from helper import Helper
from kalman_filter import KalmanFilter
from obj_tracker import MultiObjectTracker

font, font_scale, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, (57,255,20),2

device = "cuda" if torch.cuda.is_available() else "cpu"
# text_labels = [["rover", "robot", "occluded rover", "moving rover"]]
# text_labels = [['rover','many rovers']]
text_labels = [['rover']]

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', default=f'{code_dir}/../OmniLRS/scripts/nhi/collected_data/rover*', type=str)
parser.add_argument('--object_detector_id', default="iSEE-Laboratory/llmdet_tiny", type=str)
parser.add_argument('--object_detector_thres', default=0.4, type=int)
parser.add_argument('--intrinsic_file', default=f'{code_dir}/../FoundationStereo/assets/K.txt', type = str)
parser.add_argument('--max_lost_frames', default=100,type=int)
parser.add_argument('--iou_threshold',default = 0.3, type =int)

# "iSEE-Laboratory/llmdet_large", "iSEE-Laboratory/llmdet_tiny"
args = parser.parse_args()

### LlmDet for object detection
obj_det_processor = AutoProcessor.from_pretrained(args.object_detector_id)
object_det_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.object_detector_id).to(device)

### FoundationStereo model
depth_model = get_depth_estimator()
depth_estimator_args = get_args()

### Helper initialization (distance calculate)
helper = Helper(args=args)
# model = get_depth_estimator()
# print(model)
print(args.test_dir)

### obj tracker
tracker = MultiObjectTracker(max_lost_frames=args.max_lost_frames, iou_threshold=args.iou_threshold)

for dir in glob.glob(args.test_dir)[6:]: 
    print("Loading ...", dir)
    
    output_dir=f"./demo_res_2/" + dir.split('/')[-1]
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)

    # stereo data
    left_images = natsorted(glob.glob(f"{dir}/rgb_left/*.jpg"))
    right_images = natsorted(glob.glob(f"{dir}/rgb_right/*.jpg"))
    # print(len(left_images), len(right_images))
    l, r = 0 , 0
    # for i in range(min(len(left_images), len(right_images))):
    bbox_results = []
    while l < len(left_images) and r < len(right_images):
        # print(left_images[i])
        # if os.path.basename(left_images[i]) != os.path.basename(right_images[i]):
        #     continue
        # print(os.path.basename(left_images[l]))
        number_left = int(re.search(r'\d+', os.path.basename(left_images[l])).group())
        number_right = int(re.search(r'\d+', os.path.basename(right_images[r])).group())
        if number_left < number_right:
            l += 1
            continue
        elif number_right < number_left:
            r+=1
            continue
        else: # number_right = number_left => foundationstereo
            # img0 = imageio.imread(args.left_file)
            # img1 = imageio.imread(args.right_file)
            ## ===================
            ##  Depth Estimation
            ## ===================
            img_l = imageio.imread(left_images[l])
            img_r = imageio.imread(right_images[r])
            depth = get_depth(depth_model, depth_estimator_args, img_l, img_r)
            # print(depth.shape)
            ## =====================
            ## Object Detection
            ## =====================

            inputs = obj_det_processor(images=img_l, text=text_labels, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = object_det_model(**inputs)
            
            results = obj_det_processor.post_process_grounded_object_detection(
                outputs, threshold=args.object_detector_thres, target_sizes=[(img_l.shape[0], img_l.shape[1])]  
            )[0]
            # print(results['boxes'])
            # scores, boxes, text_labels, labels = results['scores'], results['boxes'], results['text_labels'], results['labels']
            # print(boxes)
            boxes = results['boxes'].cpu().numpy().astype(int)
            indices = helper.non_max_suppression(boxes, results['scores'])
            print('keep indices', indices)
            boxes = boxes[indices]
            # print(boxes)
            # tracks = tracker.update(results['boxes'].cpu())
            # bbox_results.append(tracks)
            # 
            boxes_ = []
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = box#.cpu().numpy().astype(int)
                
                # Crop depth map to box region
                depth_region = depth[y_min:y_max, x_min:x_max]
                avg_depth = depth_region.mean() 
                boxes_.append([x_min,  y_min, x_max, y_max, avg_depth])
                #box.append(avg_depth)
            print('================')
            # print(boxes_)
            tracks = tracker.update(boxes_)
            # print('123', tracks.values())
            ## calculate 3D coordinates and save results
            _3d_coords =[]
            for k in tracks.keys():
                # print(tracks[k])
                # print(x_min+(x_max-x_min)//2, y_min+(y_max-y_min)//2, avg_depth)
                coords = helper.pixel_to_3d(tracks[k][0]+(tracks[k][2]-tracks[k][0])//2, tracks[k][1]+(tracks[k][3]-tracks[k][1])//2, tracks[k][4]) # x , y ,depth
                _3d_coords.append(coords)
                start_pt, end_pt = (int(tracks[k][0]), int(tracks[k][1])), (int(tracks[k][2]),int( tracks[k][3]))
                x, y = start_pt
                cv2.putText(img_l, "id:"+str(k), (x-20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (57,255,20),3)
                cv2.rectangle(img_l, start_pt, end_pt, (57,255,20), 2)
            print("3dd",_3d_coords)
            # Convert to proper shape (N, 3) where N is number of points
            points_matrix = np.array([p.flatten() for p in _3d_coords])
            print("Points matrix shape:", points_matrix.shape)  # Should be (3, 3)

            # Calculate pairwise distances
            distances = pdist(points_matrix, metric='euclidean')
            distance_matrix = squareform(distances)
            print('Distance matrix:', distance_matrix)
            ## ================== put text distance matrix
            start_x = 50
            start_y = 50
            row_height = 40
            col_width = 150
            matrix_size = distance_matrix.shape[0]
            for i in range(matrix_size):
                for j in range(matrix_size):
                    text = f"{distance_matrix[i, j]:.4f}"
                    x = start_x + j * col_width
                    y = start_y + i * row_height
                    cv2.putText(img_l, text, (x, y), font, font_scale, color, thickness)
            # =============
            # depth_img =  cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            vis = np.hstack((cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB), depth_colored))
            # vis = np.concatenate((img_l, depth), axis=1)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(left_images[l])), vis)
            l+=1
            r+=1

            
            




