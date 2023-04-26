import cv2
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import time
import supervision as sv


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "Segment_Anything/checkpoint/sam_vit_b_01ec64.pth"



image = cv2.imread('Images/depth_image.png')

start_time = time.time()
sam = sam_model_registry[MODEL_TYPE](checkpoint= CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
print(f"time to run was {time.time()- start_time} s")

mask_annotator = sv.MaskAnnotator()
detections = sv.Detections.from_sam(masks)
annotated_image = mask_annotator.annotate(image, detections)

cv2.imshow('image window', annotated_image)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()