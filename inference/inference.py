import os, sys, glob, argparse, re, imageio
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from FoundationStereo.scripts.run_demo import *
from natsort import natsorted
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection




device = "cuda" if torch.cuda.is_available() else "cpu"
# text_labels = [["rover", "robot", "occluded rover", "moving rover"]]
text_labels = [['rover']]

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', default=f'{code_dir}/../OmniLRS/scripts/nhi/collected_data/rover*', type=str)
parser.add_argument('--object_detector_id', default="iSEE-Laboratory/llmdet_tiny", type=str)
parser.add_argument('--object_detector_thres', default=0.4, type=int)
# "iSEE-Laboratory/llmdet_large", "iSEE-Laboratory/llmdet_tiny"
args = parser.parse_args()

### LlmDet for object detection
obj_det_processor = AutoProcessor.from_pretrained(args.object_detector_id)
object_det_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.object_detector_id).to(device)

### FoundationStereo model
depth_model = get_depth_estimator()
depth_estimator_args = get_args()

# model = get_depth_estimator()
# print(model)
print(args.test_dir)

for dir in glob.glob(args.test_dir)[:1]: 
    print("Loading ...", dir)
    output_dir=f"./demo_res/"
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)

    # stereo data
    left_images = natsorted(glob.glob(f"{dir}/rgb_left/*.jpg"))
    right_images = natsorted(glob.glob(f"{dir}/rgb_right/*.jpg"))
    # print(len(left_images), len(right_images))
    l, r = 0 , 0
    # for i in range(min(len(left_images), len(right_images))):
    while l < len(left_images) or r < len(right_images):
        # print(left_images[i])
        # if os.path.basename(left_images[i]) != os.path.basename(right_images[i]):
        #     continue
        # print(os.path.basename(left_images[i]))
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
            print(depth.shape)
            ## =====================
            ## Object Detection
            ## =====================

            inputs = obj_det_processor(images=img_l, text=text_labels, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = object_det_model(**inputs)
            
            results = obj_det_processor.post_process_grounded_object_detection(
                outputs, threshold=args.object_detector_thres, target_sizes=[(img_l.shape[0], img_l.shape[1])]  
            )[0]
            print(results['boxes'])
            # scores, boxes, text_labels, labels = results['scores'], results['boxes'], results['text_labels'], results['labels']
            # print(boxes)
            for i, box in enumerate(results['boxes']):
                x_min, y_min, x_max, y_max = box.cpu().numpy().astype(int)
                
                # Crop depth map to box region
                depth_region = depth[y_min:y_max, x_min:x_max]
                
                print(depth_region.shape)
            
            




