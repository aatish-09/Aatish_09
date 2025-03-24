
import os
import cv2
import numpy as np
from tqdm import tqdm

# Disable Albumentations update check
os.environ['ALBUMENTATIONS_DISABLE_UPDATE_CHECK'] = '1'
import albumentations as A

# Define transformations with YOLO format support
transforms = [
    # Transform 1: Crop + Flip + Brightness
    A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=1),
        A.RandomBrightnessContrast(p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])),

    # Transform 2: CLAHE + ChannelDropout
    A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
        A.ChannelDropout(channel_drop_range=(1, 1), p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])),

    # Transform 3: Blur + ChannelShuffle
    A.Compose([
        A.Blur(blur_limit=(3, 3), p=1),
        A.ChannelShuffle(p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])),

    # Transform 4: GaussianBlur + ColorJitter
    A.Compose([
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0), p=1),
        A.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.3, hue=0.5, p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])),

    # Transform 5: MedianBlur + FancyPCA
    A.Compose([
        A.MedianBlur(blur_limit=3, p=1),
        A.FancyPCA(alpha=0.1, p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])),

    # Transform 6: MotionBlur + Sepia
    A.Compose([
        A.MotionBlur(blur_limit=3, p=1),
        A.ToSepia(p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])),

    # Transform 7: GlassBlur + RGBShift
    A.Compose([
        A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, mode='fast', p=1),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])),

    # Transform 8: Resize + VerticalFlip + ISONoise
    A.Compose([
        A.Resize(width=320, height=320),
        A.VerticalFlip(p=1),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])),

    # Transform 9: Resize + Rotate + GaussNoise
    A.Compose([
        A.Resize(width=352, height=352, p=1),
        A.RandomRotate90(p=1),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])),

    # Transform 10: Equalize + HSVShift + ISONoise
    A.Compose([
        A.Equalize(mode='cv', by_channels=True, p=1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))
]

def validate_bboxes(bboxes):
    """Ensure bounding boxes stay within valid range"""
    valid_boxes = []
    for bbox in bboxes:
        try:
            x, y, w, h = bbox
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            w = np.clip(w, 0.0, 1.0 - x)
            h = np.clip(h, 0.0, 1.0 - y)
            if w > 0.001 and h > 0.001:
                valid_boxes.append((x, y, w, h))
        except:
            pass
    return valid_boxes

def process_dataset():
    """Process and save augmented images with labels"""
    input_img_dir = "coco128/images/train2017"
    input_label_dir = "coco128/labels/train2017"
    output_img_dir = "coco128_augmented/images/train2017"
    output_label_dir = "coco128_augmented/labels/train2017"

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_img_dir) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in tqdm(image_files, desc="Augmenting"):
        for transform_idx, transform in enumerate(transforms, 1):
            img_path = os.path.join(input_img_dir, img_file)
            label_path = os.path.join(input_label_dir, img_file.replace('.jpg', '.txt'))
            
            # Load image
            try:
                image = cv2.imread(img_path)
                if image is None: continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                continue

            # Load labels
            bboxes, class_ids = [], []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                class_id = int(parts[0])
                                bbox = list(map(float, parts[1:5]))
                                validated = validate_bboxes([bbox])
                                if validated:
                                    bboxes.append(validated[0])
                                    class_ids.append(class_id)
                            except:
                                continue

            if not bboxes: continue

            # Apply augmentation
            try:
                transformed = transform(image=image, bboxes=bboxes, class_ids=class_ids)
            except Exception as e:
                print(f"Transform {transform_idx} failed: {str(e)}")
                continue

            # Save results
            output_img_path = os.path.join(output_img_dir, f"aug{transform_idx}_{img_file}")
            cv2.imwrite(output_img_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
            
            output_label_path = os.path.join(output_label_dir, f"aug{transform_idx}_{img_file.replace('.jpg', '.txt')}")
            with open(output_label_path, 'w') as f:
                for bbox, cls_id in zip(transformed['bboxes'], transformed['class_ids']):
                    f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

def visualize_augmentations():
    """Interactive visualization with proper BGR handling"""
    output_img_dir = "coco128_augmented/images/train2017"
    output_label_dir = "coco128_augmented/labels/train2017"
    
    aug_images = sorted([f for f in os.listdir(output_img_dir) 
                        if f.startswith('aug') and f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if not aug_images:
        print("No augmented images found! Run processing first.")
        return

    current_idx = 0
    total_images = len(aug_images)
    
    cv2.namedWindow('Augmentation Preview', cv2.WINDOW_NORMAL)
    
    while True:
        img_file = aug_images[current_idx]
        img_path = os.path.join(output_img_dir, img_file)
        label_path = os.path.join(output_label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Load image in BGR format
        image = cv2.imread(img_path)
        if image is None:
            current_idx = (current_idx + 1) % total_images
            continue
            
        h, w = image.shape[:2]
        
        # Load and draw bounding boxes
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Convert YOLO to pixel coordinates
                            x = int((x_center - width/2) * w)
                            y = int((y_center - height/2) * h)
                            x2 = int(x + width * w)
                            y2 = int(y + height * h)
                            
                            # Ensure coordinates are within image bounds
                            x, y = max(0, x), max(0, y)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            # Draw on BGR image
                            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 2)
                        except Exception as e:
                            print(f"Bbox error: {str(e)}")
                            continue

        # Resize and add UI elements
        display_img = cv2.resize(image, (800, 600))
        cv2.putText(display_img, f"Image {current_idx+1}/{total_images}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_img, "N: Next  P: Previous  Q: Quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Augmentation Preview', display_img)
        
        key = cv2.waitKey(0)
        if key == ord('n'):
            current_idx = (current_idx + 1) % total_images
        elif key == ord('p'):
            current_idx = (current_idx - 1) % total_images
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_dataset()
    visualize_augmentations()
