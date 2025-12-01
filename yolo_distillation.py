'''yolo knowledge distillation for human detection
- teacher: rt-detr/detectron2/groundingdino -> student: yolov8n (nano)
- binary classification: human vs. not human
- only the highest confidence bounding box (argmax) is kept per image'''

import os
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO, RTDETR
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm
import random
import warnings
import urllib.request

# suppress pkg_resources deprecation warning from detectron2
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*', category=UserWarning)

        # optional imports for detectron2 and groundingdino
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

try:
    from groundingdino.util.inference import load_model, predict
    from PIL import Image
    import groundingdino
    GROUNDINGDINO_AVAILABLE = True
except ImportError:
    GROUNDINGDINO_AVAILABLE = False


class YOLOHumanDistillation:
    def __init__(self, input_dir='input', output_dir='modeling', teacher_model='rtdetr-x.pt'):
        # core paths for reading raw pngs and writing all intermediate artifacts
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.teacher_model_type = None  # 'ultralytics', 'detectron2', or 'groundingdino'

        # teacher model - rt-detr (transformer-based, very accurate) by default
        # options:
        #   ultralytics: rtdetr-x.pt, rtdetr-l.pt, yolov9e.pt, yolov10x.pt, yolov8x.pt
        #   detectron2: detectron2
        #   groundingdino: groundingdino
        # teacher is responsible for generating pseudo-labels; student just learns from them
        print(f'Loading teacher model ({teacher_model})...')

        # detectron2
        if teacher_model == 'detectron2':
            if not DETECTRON2_AVAILABLE:
                raise ImportError('Detectron2 not available.')
            # standard detectron2 config from model zoo
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
            cfg.MODEL.DEVICE = 'cpu'
            self.teacher = DefaultPredictor(cfg)
            self.teacher_model_type = 'detectron2'
            print('Successfully loaded Detectron2')

        # grounding dino
        elif teacher_model == 'groundingdino':
            if not GROUNDINGDINO_AVAILABLE:
                raise ImportError('GroundingDINO not available. Install from: https://github.com/IDEA-Research/GroundingDINO')
            try:
                # find config file in installed package
                import groundingdino
                import os
                import urllib.request

                config_dir = os.path.join(os.path.dirname(groundingdino.__file__), 'config')
                config_file = os.path.join(config_dir, 'GroundingDINO_SwinB_cfg.py')

                if not os.path.exists(config_file):
                    raise FileNotFoundError(f'Config file not found: {config_file}')

                # checkpoint file - try to find it locally first, otherwise download once
                checkpoint_name = 'groundingdino_swinb_cogcoor.pth'
                checkpoint_url = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swinb_cogcoor.pth'

                # get script directory (model_training folder)
                script_dir = os.path.dirname(os.path.abspath(__file__))

                # try multiple locations for checkpoint (use absolute paths)
                possible_locations = [
                    os.path.join(script_dir, checkpoint_name),  # script directory (model_training)
                    os.path.abspath(checkpoint_name),  # current working directory
                    os.path.expanduser(f'~/{checkpoint_name}'),  # home directory
                    os.path.join(os.path.dirname(groundingdino.__file__), checkpoint_name),  # package directory
                ]

                # try to find the checkpoint file in the possible locations
                checkpoint_file = None
                for loc in possible_locations:
                    # get the absolute path of the location
                    abs_loc = os.path.abspath(loc)
                    # if the location exists, set the checkpoint file to the absolute path
                    if os.path.exists(abs_loc):
                        checkpoint_file = abs_loc
                        print(f'  Found checkpoint at: {checkpoint_file}')
                        break # break out of the loop

                # download if not found
                # if the checkpoint file is not found, download it from the GitHub repository
                if checkpoint_file is None:
                    print('Checkpoint not found. Downloading from GitHub...')
                    checkpoint_file = os.path.expanduser(f'~/{checkpoint_name}')
                    try:
                        print(f'  Downloading to: {checkpoint_file}')
                        urllib.request.urlretrieve(checkpoint_url, checkpoint_file)
                        print('  Download complete!')
                    except Exception as download_error:
                        raise FileNotFoundError(
                            f'Could not download checkpoint. Error: {download_error}\n'
                            f'Please download manually from: {checkpoint_url}\n'
                            f'and place it in: {checkpoint_file}'
                        )

                # determine device - use cpu for groundingdino (mps not fully supported)
                if torch.cuda.is_available():
                    device = 'cuda'
                else:
                    device = 'cpu'

                print(f'  Loading model on device: {device} (GroundingDINO uses CPU on Mac)')
                self.teacher = load_model(config_file, checkpoint_file, device=device)
            except Exception as e:
                raise ImportError(
                    f'Could not load GroundingDINO. Error: {e}. '
                    f'Make sure config file exists and checkpoint is available.'
                )
            self.teacher_model_type = 'groundingdino'
            print('Successfully loaded GroundingDINO')

        # ultralytics models (rt-detr or yolo)
        else:
            try:
                if teacher_model.startswith('rtdetr'):
                    self.teacher = RTDETR(teacher_model)
                    self.teacher_model_type = 'ultralytics'
                    print(f'Successfully loaded RT-DETR model: {teacher_model}')
                else:
                    self.teacher = YOLO(teacher_model)
                    self.teacher_model_type = 'ultralytics'
                    print(f'Successfully loaded YOLO model: {teacher_model}')
            except Exception as e:
                print(f'Warning: Could not load {teacher_model}, trying RT-DETR-L...')
                try:
                    self.teacher = RTDETR('rtdetr-l.pt')
                    self.teacher_model_type = 'ultralytics'
                    print('Fallback to RT-DETR-L')
                except Exception as e2:
                    print('Warning: Could not load RT-DETR-L, trying YOLOv8x...')
                    try:
                        self.teacher = YOLO('yolov8x.pt')
                        self.teacher_model_type = 'ultralytics'
                        print('Fallback to YOLOv8x')
                    except Exception as e3:
                        print(f'Error: Could not load any teacher model: {e3}')
                        raise

        # student model (nano)
        print('Loading student model (YOLOv8n)...')
        self.student = YOLO('yolov8n.pt')

        # coco class 0 is 'person'
        self.human_class_id = 0

        # confidence threshold for teacher (lowered for better detection)
        self.teacher_conf_threshold = 0

        # split ratios used when taking raw pngs and making train/val/test subsets
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

    def split_images(self):
        """
        split input images into train/val/test (80/10/10)
        - only cares about raw pngs; labels come later from the teacher
        """
        print('\n' + '-' * 60)
        print('Step 1: Splitting images into train/val/test')
        print('-' * 60)

        # get all png images in the configured input directory
        input_images = list(self.input_dir.glob('*.png'))

        if not input_images:
            raise ValueError(f'No PNG images found in {self.input_dir}')

        print(f'Found {len(input_images)} PNG images')

        # shuffle once so train/val/test are random but reproducible via seed
        random.shuffle(input_images)

        # calculate split indices
        total = len(input_images)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_images = input_images[:train_end]
        val_images = input_images[train_end:val_end]
        test_images = input_images[val_end:]

        # create split directories under the main output directory
        split_dir = self.output_dir / 'split'
        for split in ['train', 'val', 'test']:
            (split_dir / split).mkdir(parents=True, exist_ok=True)

        # copy images to respective directories (no labels yet)
        print('\nCopying images to split directories...')
        for img_path in tqdm(train_images, desc='Train'):
            shutil.copy(img_path, split_dir / 'train' / img_path.name)

        for img_path in tqdm(val_images, desc='Val'):
            shutil.copy(img_path, split_dir / 'val' / img_path.name)

        for img_path in tqdm(test_images, desc='Test'):
            shutil.copy(img_path, split_dir / 'test' / img_path.name)

        print('\nSplit complete:')
        print(f'  Train: {len(train_images)} images ({len(train_images) / total * 100:.1f}%)')
        print(f'  Val:   {len(val_images)} images ({len(val_images) / total * 100:.1f}%)')
        print(f'  Test:  {len(test_images)} images ({len(test_images) / total * 100:.1f}%)')

        return split_dir

    def detect_humans(self, img_path, confidence_threshold):
        """
        unified detection method that works with all model types
        - returns only the highest confidence bounding box (argmax)
        - normalizes outputs into a single common yolo-style format
        - returns: dict with 'bbox', 'bbox_abs', 'confidence', or none if no detections
        """
        # read the image
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        # get the height and width of the image
        img_h, img_w = img.shape[:2]
        human_boxes = [] # initialize the human boxes list

        # if the teacher model type is ultralytics, use the ultralytics api
        if self.teacher_model_type == 'ultralytics':
            # rt-detr or yolo models using the ultralytics api
            results = self.teacher(
                str(img_path),
                conf=confidence_threshold,
                iou=0.45,
                imgsz=1280,
                verbose=False,
            )

            # iterate over the results
            for result in results:
                boxes = result.boxes
                # iterate over the boxes
                for box in boxes:
                    # get the class id
                    class_id = int(box.cls[0])
                    if class_id == self.human_class_id:
                        # convert from absolute xyxy into normalized center/width/height
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = ((x1 + x2) / 2) / img_w
                        center_y = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        confidence = float(box.conf[0])

                        # append the human box to the list
                        human_boxes.append(
                            {
                                'bbox': [center_x, center_y, width, height],
                                'bbox_abs': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                            }
                        )

        # if the teacher model type is detectron2, use the detectron2 api
        elif self.teacher_model_type == 'detectron2':
            # detectron2 (coco class 0 is person)
            outputs = self.teacher(img)
            instances = outputs['instances']

            # iterate over the instances
            for i in range(len(instances)):
                # get the class id
                class_id = instances.pred_classes[i].item()
                # if the class id is the human class id, append the human box to the list
                if class_id == self.human_class_id:
                    # get the score
                    score = instances.scores[i].item()
                    # if the score is greater than the confidence threshold, append the human box to the list
                    if score >= confidence_threshold:
                        # get the box
                        box = instances.pred_boxes[i].tensor[0].cpu().numpy()
                        # get the x1, y1, x2, y2 coordinates from the box
                        x1, y1, x2, y2 = box
                        # get the center x and y coordinates
                        center_x = ((x1 + x2) / 2) / img_w
                        center_y = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h

                        # append the human box to the list
                        human_boxes.append(
                            {
                                'bbox': [center_x, center_y, width, height],
                                'bbox_abs': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': score,
                            }
                        )

        # if the teacher model type is groundingdino, use the groundingdino api
        elif self.teacher_model_type == 'groundingdino':
            # groundingdino with prompt tuned for noisy, low-res images
            text_prompt = 'person . human . man standing . people .'
            # open the image and convert it to a PIL image - errors otherwise
            pil_image = Image.open(img_path).convert('RGB')

            # create a transform to convert the image to a tensor
            transform = T.Compose(
                [
                    T.ToTensor(),
                ]
            )
            # convert the image to a tensor using the transform
            image_tensor = transform(pil_image)

            # determine the device to use for the groundingdino model
            if torch.backends.mps.is_available():
                gdino_device = 'cpu' # use the cpu for the groundingdino model on mac - unsupported on mps
            elif torch.cuda.is_available():
                gdino_device = 'cuda'
            else:
                gdino_device = 'cpu'

            # predict the boxes, logits, and phrases using the groundingdino model
            boxes, logits, phrases = predict(
                model=self.teacher,
                image=image_tensor,
                caption=text_prompt, # the prompt to use for the groundingdino model
                box_threshold=confidence_threshold, # the confidence threshold for the boxes
                text_threshold=0.25, # the confidence threshold for the text
                device=gdino_device,
            )

            # groundingdino returns boxes in normalized format [x_center, y_center, width, height]
            # iterate over the boxes and logits and append the human box to the list if the logit is greater than the confidence threshold
            for i, (box, logit) in enumerate(zip(boxes, logits)):
                if logit >= confidence_threshold:
                    x_center, y_center, w, h = box
                    # get the x1, y1, x2, y2 coordinates from the box
                    x1 = (x_center - w / 2) * img_w
                    y1 = (y_center - h / 2) * img_h
                    x2 = (x_center + w / 2) * img_w
                    y2 = (y_center + h / 2) * img_h

                    # append the human box to the list
                    human_boxes.append(
                        {
                            'bbox': [x_center, y_center, w, h],
                            'bbox_abs': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(logit),
                        }
                    )

        # if there are any human boxes, return the best box
        if human_boxes:
            # return the best box - the one with the highest confidence
            best_box = max(human_boxes, key=lambda x: x['confidence'])
            return best_box
        # if there are no human boxes, return None
        return None

    def generate_labels_from_teacher(self, split_dir, confidence_threshold=None):
        """
        use teacher model to generate bounding box labels for all splits
        - outputs:
          - model_readable/: yolo format labels for training
          - human_readable/: images with bounding boxes drawn (for quick scans)
          - comparison/: side-by-side original vs labeled (for debugging)
        """
        # if the confidence threshold is not provided, use the default confidence threshold
        if confidence_threshold is None:
            confidence_threshold = self.teacher_conf_threshold

        print('\n' + '-' * 60)
        print('Step 2: Generating labels from teacher model')
        print('-' * 60)

        # where model-readable images + labels will go
        model_readable_dir = self.output_dir / 'model_readable'
        # where human-readable images with bounding boxes drawn for quick inspection will go
        human_readable_dir = self.output_dir / 'human_readable'
        # where side-by-side original vs labeled images will go
        comparison_dir = self.output_dir / 'comparison'

        stats = {}

        # iterate over each dataset split independently so stats are per-split
        for split in ['train', 'val', 'test']:
            # print the progress
            print(f'\nProcessing {split} set...')

            # where training-ready images + labels will go
            model_images_out = model_readable_dir / split / 'images'
            model_labels_out = model_readable_dir / split / 'labels'
            # create the directories if they don't exist
            model_images_out.mkdir(parents=True, exist_ok=True)
            model_labels_out.mkdir(parents=True, exist_ok=True)

            # human-readable images with bounding boxes drawn for quick inspection
            human_images_out = human_readable_dir / split
            human_images_out.mkdir(parents=True, exist_ok=True)

            # side-by-side original vs labeled composite
            comparison_images_out = comparison_dir / split
            comparison_images_out.mkdir(parents=True, exist_ok=True)

            # raw pngs that belong to this split
            split_images = list((split_dir / split).glob('*.png'))

            # initialize the counters
            processed_count = 0
            skipped_count = 0
            human_detected_count = 0
            total_humans = 0

            # iterate over the images in the split
            for img_path in tqdm(split_images, desc=f'  {split.capitalize()}'):
                # get the name of the image
                img_name = img_path.name
                # get the destination image path for the model-readable images
                model_dst_img = model_images_out / img_name
                # get the label name for the model-readable labels
                label_name = img_path.stem + '.txt'
                model_label_path = model_labels_out / label_name
                # get the destination image path for the human-readable images
                human_dst_img = human_images_out / img_name
                # get the destination image path for the side-by-side original vs labeled images
                comparison_dst_img = comparison_images_out / img_name

                # resume capability: if all artifacts already exist, don't recompute
                # if the model-readable image, label, human-readable image, and side-by-side original vs labeled image already exist, skip the image
                if (
                    model_dst_img.exists()
                    and model_label_path.exists()
                    and human_dst_img.exists()
                    and comparison_dst_img.exists()
                ):
                    # increment the skipped count
                    skipped_count += 1
                    # if the label is not empty, increment the human detected count and total humans count
                    if model_label_path.stat().st_size > 0:
                        human_detected_count += 1
                        total_humans += 1
                    # increment the processed count
                    processed_count += 1
                    # continue to the next image
                    continue

                # run teacher once per image to get the best human box (if any)
                best_box = self.detect_humans(img_path, confidence_threshold)

                # read the image
                img = cv2.imread(str(img_path))
                # if the image is not loaded, print a warning and continue to the next image
                if img is None:
                    print(f'Warning: Could not load image {img_path}')
                    continue

                img_original = img.copy()  # keep a pristine copy for comparison view

                if best_box:
                    # draw green rectangle + confidence label for the detected human
                    x1, y1, x2, y2 = best_box['bbox_abs']
                    confidence = best_box['confidence']

                    # draw a green rectangle around the detected human
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # create the label for the detected human
                    label = f'Human {confidence:.2f}'

                    # get the size of the label
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        img,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 255, 0),
                        -1,
                    )
                    # draw the label text
                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

                # training images mirror the original input exactly (no drawings)
                shutil.copy(img_path, model_dst_img)

                # if the best box is not None, increment the human detected count and total humans count
                if best_box:
                    # increment the human detected count
                    human_detected_count += 1
                    # increment the total humans count
                    total_humans += 1
                    # open the model label path and write the bounding box to the file
                    with open(model_label_path, 'w') as f:
                        # get the bounding box
                        bbox = best_box['bbox']
                        # write the bounding box to the file 
                        f.write(f'0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n')
                else:
                    # if the best box is None, create an empty label file
                    open(model_label_path, 'w').close()

                # write the image to the human-readable image directory
                cv2.imwrite(str(human_dst_img), img)

                # make two versions of the image for the side-by-side panel:
                # one with the filename and one with the bounding box
                img_orig_with_text = img_original.copy()   # plain + filename
                img_annotated_with_text = img.copy()       # labeled + filename

                # get the filename text
                filename_text = img_name
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_color = (255, 255, 255)
                bg_color = (0, 0, 0)

                # get text size for filename
                (text_width, text_height), baseline = cv2.getTextSize(
                    filename_text,
                    font,
                    font_scale,
                    font_thickness,
                )


                # draw a rectangle around the filename text
                cv2.rectangle(
                    img_orig_with_text,
                    (10, 10),
                    (10 + text_width + 10, 10 + text_height + baseline + 10),
                    bg_color,
                    -1,
                )

                # draw filename text
                cv2.putText(
                    img_orig_with_text,
                    filename_text,
                    (15, 10 + text_height),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                )

                (text_width, text_height), baseline = cv2.getTextSize(
                    filename_text,
                    font,
                    font_scale,
                    font_thickness,
                )

                cv2.rectangle(
                    img_annotated_with_text,
                    (10, 10),
                    (10 + text_width + 10, 10 + text_height + baseline + 10),
                    bg_color,
                    -1,
                )

                cv2.putText(
                    img_annotated_with_text,
                    filename_text,
                    (15, 10 + text_height),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                )

                # horizontally concatenate original + annotated for easier review
                # get the height and width of the original image
                h, w = img_original.shape[:2]
                # concatenate the original and annotated images horizontally to create the comparison image
                comparison_img = np.hstack([img_orig_with_text, img_annotated_with_text])

                # set the label height
                label_height = 40
                # allocate white strip at the top for "original" / "labeled" titles to create the comparison with labels image
                comparison_with_labels = np.ones((h + label_height, w * 2, 3), dtype=np.uint8) * 255
                comparison_with_labels[label_height:, :] = comparison_img

                # draw original label for the comparison image
                original_label = 'Original'
                (label_w, label_h), _ = cv2.getTextSize(original_label, font, 0.8, 2)
                cv2.putText(
                    comparison_with_labels,
                    original_label,
                    (w // 2 - label_w // 2, 30),
                    font,
                    0.8,
                    (0, 0, 0),
                    2,
                )

                # draw labeled label for the comparison image
                labeled_label = 'Labeled'
                (label_w, label_h), _ = cv2.getTextSize(labeled_label, font, 0.8, 2)
                cv2.putText(
                    comparison_with_labels,
                    labeled_label,
                    (w + w // 2 - label_w // 2, 30),
                    font,
                    0.8,
                    (0, 0, 0),
                    2,
                )

                # save comparison image
                cv2.imwrite(str(comparison_dst_img), comparison_with_labels)

                processed_count += 1

            # update stats
            stats[split] = {
                'total': processed_count,
                'skipped': skipped_count,
                'with_humans': human_detected_count,
                'without_humans': processed_count - human_detected_count,
                'total_humans': total_humans,
                'percent_with_humans': (human_detected_count / processed_count * 100) if processed_count > 0 else 0,
            }

            # print stats
            print(f'\n  {split.capitalize()} set processed:')
            print(f'    Total images: {processed_count}')
            if skipped_count > 0:
                print(f'    Skipped (already processed): {skipped_count}')
            print(f'    Images with humans: {human_detected_count} ({stats[split]["percent_with_humans"]:.1f}%)')
            print(f'    Images without humans: {stats[split]["without_humans"]}')
            print(f'    Total human detections: {total_humans}')

        # summary
        print('\n' + '-' * 60)
        print('Teacher Model Labeling Summary')
        print('-' * 60)
        total_all = sum(s['total'] for s in stats.values())
        total_with_humans = sum(s['with_humans'] for s in stats.values())
        total_detections = sum(s['total_humans'] for s in stats.values())

        print('\nOverall:')
        print(f'  Total images: {total_all}')
        print(f'  Images with humans: {total_with_humans} ({total_with_humans / total_all * 100:.1f}%)')
        print(f'  Total human detections: {total_detections}')
        print(f'  Average humans per image: {total_detections / total_with_humans:.2f}')

        print('\nOutput directories:')
        print(f'  Model-readable (YOLO format): {model_readable_dir}')
        print(f'  Human-readable (annotated): {human_readable_dir}')
        print(f'  Comparison (side-by-side): {comparison_dir}')

        return model_readable_dir, stats

    def review_and_edit_labels(self, split='train'):
        """
        interactive tool to review and edit labels
        - supports drawing bounding boxes and removing images
        - saves results to a new directory to preserve originals
        """
        # print header with instructions
        print('\n' + '-' * 60)
        print(f'Interactive Label Review Tool - {split.upper()} Set')
        print('-' * 60)
        print('\nControls:')
        print('  Click & Drag: Draw new bounding box (always enabled)')
        print("  'w' or Up Arrow: Next image (forward)")
        print("  's' or Down Arrow: Previous image (back)")
        print("  'd' key: Delete current image")
        print("  'a' key: Remove all bounding boxes from current image")
        print("  'q' or ESC: Quit and save all changes")
        print('-' * 60)

        # set up directories
        model_readable_dir = self.output_dir / 'model_readable'
        original_images_dir = model_readable_dir / split / 'images'
        original_labels_dir = model_readable_dir / split / 'labels'

        # set up reviewed directory
        reviewed_dir = self.output_dir / 'model_readable_reviewed'
        images_dir = reviewed_dir / split / 'images'
        labels_dir = reviewed_dir / split / 'labels'
        deleted_dir = reviewed_dir / split / 'deleted'

        # create them on the computer if they don't exist
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        deleted_dir.mkdir(parents=True, exist_ok=True)  # soft-delete bucket so nothing is lost

        print('\nCopying original labels to reviewed directory...')
        original_images = list(original_images_dir.glob('*.png'))
        for img_path in tqdm(original_images, desc='  Copying'):
            # copy image to reviewed directory
            dst_img = images_dir / img_path.name
            dst_label = labels_dir / (img_path.stem + '.txt')
            src_label = original_labels_dir / (img_path.stem + '.txt')

            # copy image to reviewed directory if it doesn't exist
            if not dst_img.exists():
                shutil.copy(img_path, dst_img)

            # copy label to reviewed directory if it doesn't exist
            if not dst_label.exists() and src_label.exists():
                shutil.copy(src_label, dst_label)
            elif not dst_label.exists():
                # create an empty label file if it doesn't exist
                dst_label.touch()

        print(f'Copied to: {reviewed_dir}')

        # get list of images to review
        image_files = sorted(list(images_dir.glob('*.png')))
        if not image_files:
            print(f'No images found in {images_dir}')
            print(f'Skipping {split} set (no images to review)')
            return reviewed_dir

        # set up drawing variables
        drawing = False
        start_point = None
        current_boxes = []
        current_idx = 0  # index into image_files for the currently displayed image

        # load image and its labels
        def load_image_and_labels(idx):
            """load image and its labels"""
            # if the index is out of bounds, return None
            if idx < 0 or idx >= len(image_files):
                return None, None, None

            # get the image path
            img_path = image_files[idx]  # resolved path of the current image
            # get the label path
            label_path = labels_dir / (img_path.stem + '.txt')

            # read the image
            img_local = cv2.imread(str(img_path))
            if img_local is None:
                return None, None, None

            boxes = []
            # load labels if they exist
            if label_path.exists():
                # open label file
                with open(label_path, 'r') as f:
                    # parse each line
                    for line in f:
                        # split line into parts
                        parts = line.strip().split()
                        if len(parts) == 5:
                            # parse parts into float values
                            _, cx, cy, w, h = map(float, parts)
                            # get image height and width
                            img_h_local, img_w_local = img_local.shape[:2]
                            # convert from normalized center/width/height to pixel xyxy
                            x1_local = int((cx - w / 2) * img_w_local)
                            y1_local = int((cy - h / 2) * img_h_local)
                            x2_local = int((cx + w / 2) * img_w_local)
                            y2_local = int((cy + h / 2) * img_h_local)
                            # add box to list
                            boxes.append([x1_local, y1_local, x2_local, y2_local])

            return img_local, boxes, img_path

        def draw_boxes(img_local, boxes):
            """draw bounding boxes on image"""
            img_copy = img_local.copy()
            # draw each box
            for box in boxes:
                x1_local, y1_local, x2_local, y2_local = box
                # draw rectangle
                cv2.rectangle(img_copy, (x1_local, y1_local), (x2_local, y2_local), (0, 255, 0), 2)
            return img_copy

        def save_labels(img_path_local, boxes, img_shape):
            """save labels in yolo format"""
            # get the label path
            label_path = labels_dir / (img_path_local.stem + '.txt')
            img_h_local, img_w_local = img_shape[:2]

            # open label file
            with open(label_path, 'w') as f:
                # write each box
                for box in boxes:
                    x1_local, y1_local, x2_local, y2_local = box
                    # get image height and width
                    # convert from pixel xyxy to normalized center/width/height
                    cx = ((x1_local + x2_local) / 2) / img_w_local
                    cy = ((y1_local + y2_local) / 2) / img_h_local
                    w = (x2_local - x1_local) / img_w_local
                    h = (y2_local - y1_local) / img_h_local
                    
                    # clamp to [0, 1] to avoid any out-of-bounds values
                    # write box
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))
                    f.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')

        def mouse_callback(event, x, y, flags, param):
            """handle mouse events for drawing boxes"""
            # set up drawing variables for the mouse callback
            nonlocal drawing, start_point, current_boxes # nonlocal means the variables are not local to the function, instead to the outer scope

            # handle left button down for the mouse callback
            if event == cv2.EVENT_LBUTTONDOWN:
                # start drawing a new box
                drawing = True
                # set the start point to the current mouse position
                start_point = (x, y)

            # handle mouse move
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing and start_point:
                    img_local, _, _ = load_image_and_labels(current_idx)
                    if img_local is not None:
                        display_img_local = draw_boxes(img_local, current_boxes)
                        cv2.rectangle(display_img_local, start_point, (x, y), (255, 0, 0), 2)
                        cv2.imshow('Label Editor', display_img_local)

            # handle when the left button is released
            elif event == cv2.EVENT_LBUTTONUP:
                if drawing and start_point:
                    # stop drawing
                    drawing = False
                    # set the end point to the current mouse position
                    end_point = (x, y)
                    x1_local, y1_local = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
                    x2_local, y2_local = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])

                    # enforce a minimum box size to avoid accidental clicks (10 pixels)
                    if abs(x2_local - x1_local) > 10 and abs(y2_local - y1_local) > 10:
                        current_boxes.append([x1_local, y1_local, x2_local, y2_local])
                        img_local, _, _ = load_image_and_labels(current_idx)
                        # if the image is not None, draw the boxes
                        if img_local is not None:
                            display_img_local = draw_boxes(img_local, current_boxes)
                            cv2.imshow('Label Editor', display_img_local)

        # load first image and its labels
        img, current_boxes, img_path = load_image_and_labels(current_idx)
        if img is None:
            print('Could not load first image')
            return reviewed_dir

        # set up display window
        display_img = draw_boxes(img, current_boxes)
        cv2.namedWindow('Label Editor', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Label Editor', mouse_callback)

        # print review info
        print(f'\nReviewing {len(image_files)} images')
        print(f'Image {current_idx + 1}/{len(image_files)}: {img_path.name}')

        # main review loop
        while True:
            # overlay simple text hud with current index + filename
            info_img = display_img.copy()
            info_text = (
                f'Image {current_idx + 1}/{len(image_files)} | Boxes: {len(current_boxes)} | {img_path.name}'
            )
            # overlay simple text hud with current index + filename
            cv2.putText(
                info_img,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            # overlay simple text hud with current index + filename
            cv2.putText(
                info_img,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
            )
            # show info image
            cv2.imshow('Label Editor', info_img)

            # wait for key press
            key_code = cv2.waitKey(1)
            # get key code
            key = key_code & 0xFF

            # handle quit key
            if key == ord('q') or key == 27:
                # quit and persist whatever boxes are on the current image
                save_labels(img_path, current_boxes, img.shape)
                break
            # handle next image key
            elif key == ord('w') or key_code == 82 or key_code == 65362:
                # next image (forward)
                save_labels(img_path, current_boxes, img.shape)
                current_idx = min(current_idx + 1, len(image_files) - 1)
                img, current_boxes, img_path = load_image_and_labels(current_idx)
                if img is not None:
                    display_img = draw_boxes(img, current_boxes)
                    print(f'Image {current_idx + 1}/{len(image_files)}: {img_path.name}')
            
            # handle previous image key
            elif key == ord('s') or key_code == 84 or key_code == 65364:
                # previous image (back)
                save_labels(img_path, current_boxes, img.shape)
                current_idx = max(current_idx - 1, 0)
                img, current_boxes, img_path = load_image_and_labels(current_idx)
                if img is not None:
                    display_img = draw_boxes(img, current_boxes)
                    print(f'Image {current_idx + 1}/{len(image_files)}: {img_path.name}')
            
            # handle delete image key
            elif key == ord('d'):
                # delete image + its labels by moving them into the "deleted" folder
                deleted_img_path = deleted_dir / img_path.name
                deleted_label_path = deleted_dir / (img_path.stem + '.txt')
                label_path = labels_dir / (img_path.stem + '.txt')

                shutil.move(str(img_path), str(deleted_img_path))
                if label_path.exists():
                    shutil.move(str(label_path), str(deleted_label_path))

                image_files.pop(current_idx)
                if current_idx >= len(image_files):
                    current_idx = len(image_files) - 1

                if len(image_files) == 0:
                    print('All images deleted. Exiting.')
                    break

                img, current_boxes, img_path = load_image_and_labels(current_idx)
                if img is not None:
                    display_img = draw_boxes(img, current_boxes)
                    print(f'Deleted. Image {current_idx + 1}/{len(image_files)}: {img_path.name}')
            
            # handle clear all boxes key
            elif key == ord('a'):
                # clear all boxes for this image but keep the image itself
                current_boxes.clear()
                save_labels(img_path, current_boxes, img.shape)
                display_img = draw_boxes(img, current_boxes)
                print(f'Removed all bounding boxes from {img_path.name}')

        # destroy all windows
        cv2.destroyAllWindows()

        # save labels if image is not None
        if img is not None:
            save_labels(img_path, current_boxes, img.shape)

        # print review complete
        print('\nReview complete!')
        print(f'  Images remaining: {len(image_files)}')
        print(f'  Deleted images moved to: {deleted_dir}')
        print(f'  Reviewed labels saved to: {reviewed_dir}')

        return reviewed_dir

    def create_dataset_yaml(self, labeled_dir):
        """create dataset.yaml for yolo training
        this is the file that ultralytics uses to train the model
        """
        dataset_yaml = {
            'path': str(labeled_dir.absolute()), # the path to the labeled directory
            'train': 'train/images', # the path to the train images
            'val': 'val/images', # the path to the validation images
            'test': 'test/images', # the path to the test images
            'nc': 1, # the number of classes
            'names': ['human'], # the names of the classes
        }

        yaml_path = labeled_dir / 'dataset.yaml' # the path to the dataset.yaml file
        with open(yaml_path, 'w') as f: # open the dataset.yaml file for writing
            yaml.dump(dataset_yaml, f, default_flow_style=False)

        print(f'\nCreated dataset.yaml at {yaml_path}')
        return yaml_path

    def train_student(self, yaml_path, epochs=100, imgsz=640, batch=16, device='mps'):
        """
        train student model on distilled dataset
        - wraps ultralytics .train with defaults tuned for this project
        """
        print('\n' + '-' * 60)
        print('Step 5: Training student model (YOLOv8n)')
        print('-' * 60)

        # all ultralytics training artifacts will live under this directory
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        # train student model
        results = self.student.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(models_dir),
            name='human_detector',
            exist_ok=True,  # do not overwrite existing project
            patience=20,
            save=True,
            verbose=True,
            hsv_h=0.015,  # hsv augmentation
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,  # spatial augmentation
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
        )

        print('\nTraining complete!')
        best_model_path = models_dir / 'human_detector' / 'weights' / 'best.pt'
        print(f'Best model saved to: {best_model_path}')

        metrics = results.results_dict  # flat dict of all logged training metrics
        train_metrics = {
            'final_epoch': len(results.results_dict) if hasattr(results, 'results_dict') else epochs,
            'box_loss': metrics.get('train/box_loss', 'N/A'),
            'cls_loss': metrics.get('train/cls_loss', 'N/A'),
            'dfl_loss': metrics.get('train/dfl_loss', 'N/A'),
        }

        return best_model_path, train_metrics

    def evaluate_student(self, model_path, yaml_path):
        """
        evaluate student model on validation and test sets
        - uses ultralytics .val for both splits and reports box metrics
        """
        print('\n' + '-' * 60)
        print('Step 6: Evaluating student model')
        print('-' * 60)

        student_model = YOLO(str(model_path))

        print('\nValidation Set Evaluation:')
        print('-' * 60)
        val_results = student_model.val(
            data=str(yaml_path),
            split='val',
        )

        val_metrics = {
            'mAP50': val_results.box.map50,
            'mAP50-95': val_results.box.map,
            'precision': val_results.box.mp,
            'recall': val_results.box.mr,
        }

        # print the validation metrics
        print(f'  Precision: {val_metrics["precision"]:.4f}')
        print(f'  Recall: {val_metrics["recall"]:.4f}')
        print(f'  mAP@0.5: {val_metrics["mAP50"]:.4f}')
        print(f'  mAP@0.5:0.95: {val_metrics["mAP50-95"]:.4f}')

        print('\nTest Set Evaluation (Final Accuracy):')
        print('-' * 60)
        test_results = student_model.val(
            data=str(yaml_path),
            split='test',
        )

        # print the test metrics
        test_metrics = {
            'mAP50': test_results.box.map50,
            'mAP50-95': test_results.box.map,
            'precision': test_results.box.mp,
            'recall': test_results.box.mr,
        }

        print(f'  Precision: {test_metrics["precision"]:.4f}')
        print(f'  Recall: {test_metrics["recall"]:.4f}')
        print(f'  mAP@0.5: {test_metrics["mAP50"]:.4f}')
        print(f'  mAP@0.5:0.95: {test_metrics["mAP50-95"]:.4f}')

        return val_metrics, test_metrics

    def visualize_predictions(self, model_path, reviewed_dir, splits=None, conf_threshold=0.25):
        """
        visualize model predictions on validation and test sets
        - saves annotated images with predicted boxes
        - draws ground truth in green and predictions in red for fast eyeballing
        """
        if splits is None:
            splits = ['val', 'test']

        print('\n' + '-' * 60)
        print('Visualizing Model Predictions')
        print('-' * 60)

        # load the model and create the predictions directory
        model = YOLO(str(model_path))
        predictions_dir = self.output_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # iterate over the splits
        for split in splits:
            print(f'\nProcessing {split.upper()} set...')
            split_images_dir = reviewed_dir / split / 'images'
            split_labels_dir = reviewed_dir / split / 'labels'
            output_dir = predictions_dir / split
            output_dir.mkdir(parents=True, exist_ok=True)

            # get the image files
            image_files = sorted(list(split_images_dir.glob('*.png')))
            if not image_files:
                print(f'  No images found in {split_images_dir}')
                continue

            # iterate over the image files
            for img_path in tqdm(image_files, desc=f'  {split.capitalize()}'):
                # if the image is not found, continue to next image
                # run inference on each reviewed image using the trained model
                results = model.predict(
                    str(img_path),
                    conf=conf_threshold,
                    imgsz=640,
                    verbose=False,
                )
                # read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue # if image is not found, continue to next image

                # get the label path
                label_path = split_labels_dir / (img_path.stem + '.txt')
                # if the label path exists, draw the ground truth boxes
                if label_path.exists():
                    # draw ground-truth boxes first (green)
                    with open(label_path, 'r') as f:
                        for line in f:
                            # parse label
                            parts = line.strip().split()
                            if len(parts) == 5:
                                _, cx, cy, w, h = map(float, parts)
                                img_h, img_w = img.shape[:2] # get image height and width
                                x1 = int((cx - w / 2) * img_w) # convert to pixel coordinates
                                y1 = int((cy - h / 2) * img_h)
                                x2 = int((cx + w / 2) * img_w)
                                y2 = int((cy + h / 2) * img_h)

                                # draw ground truth box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                # draw ground truth label
                                cv2.putText(
                                    img,
                                    'GT',
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                )

                # overlay predicted boxes (red) on top
                for result in results:
                    boxes = result.boxes # get boxes
                    for box in boxes:
                        # overlay predicted boxes (red) on top
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        # setup predicted box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # setup predicted label
                        label = f'Pred {confidence:.2f}'
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        
                        # draw predicted box
                        cv2.rectangle(
                            img,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            (0, 0, 255),
                            -1,
                        )

                        # draw predicted label
                        cv2.putText(
                            img,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                # setup legend
                legend_y = 30
                # draw ground truth legend
                cv2.putText(
                    img,
                    'Green: Ground Truth',
                    (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                # draw predictions legend
                cv2.putText(
                    img,
                    'Red: Predictions',
                    (10, legend_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

                # save annotated image
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), img)

            print(f'  Saved {len(image_files)} annotated images to {output_dir}')

        print(f'\nAll predictions saved to: {predictions_dir}')
        return predictions_dir

    @staticmethod
    def detect_human_deployed(model_path, image_path_or_array, conf_threshold=0.45, imgsz=640):
        """detect a human in an image using a deployed model"""
        model = YOLO(str(model_path))

        # run inference
        results = model.predict(
            image_path_or_array,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        # setup detections list
        detections = []
        for result in results:
            boxes = result.boxes # get boxes
            if boxes is not None and len(boxes) > 0:
                img_h, img_w = result.orig_shape[:2] # get image height and width

                for box in boxes:
                    confidence = float(box.conf[0]) # get confidence
                    if confidence > conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # convert to normalized coordinates
                        cx = ((x1 + x2) / 2) / img_w
                        cy = ((y1 + y2) / 2) / img_h
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h

                        # add detection to list
                        detections.append(
                            {
                                'bbox': [cx, cy, w, h],
                                'bbox_abs': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                            }
                        )

        # return best detection if detections are found
        if detections:
            best_detection = max(detections, key=lambda x: x['confidence'])
            return best_detection
        # return None if no detections are found
        return None

    def export_model(self, model_path, format='torchscript'):
        """
        export trained model for deployment
        - ultralytics handles format-specific details under the hood
        """
        print('\n' + '-' * 60)
        print('Step 7: Exporting model')
        print('-' * 60)

        print(f'\nExporting to {format} format...')
        model = YOLO(str(model_path))
        export_path = model.export(format=format)
        print(f'Model exported to: {export_path}')
        return export_path

    def run_full_pipeline(self, epochs=10, imgsz=640, batch=2, device='mps'):
        """run complete distillation pipeline"""
        # print header
        print('-' * 60)
        print('YOLO Human Detection Knowledge Distillation')
        teacher_model_name = getattr(self.teacher, 'model_name', 'Strong Teacher Model')
        print(f'Teacher: {teacher_model_name} -> Student: YOLOv8n')
        print('-' * 60)

        split_dir = self.split_images()

        model_readable_dir, label_stats = self.generate_labels_from_teacher(split_dir) # generate labels from teacher

        if label_stats['train']['with_humans'] == 0: # if no humans detected in training set, print error and return
            print('\nNo humans detected in training set. Cannot proceed.')
            return

        print('\n' + '-' * 60)
        print('Step 3: Interactive Label Review')
        print('-' * 60)
        print('\nYou will review all three splits (train, val, test).')
        print('You can draw bounding boxes, delete images, and edit labels.')

        reviewed_dir = None
        for split in ['train', 'val', 'test']: # review each split
            print('\n' + '-' * 60)
            print(f'Reviewing {split.upper()} set...')
            print('-' * 60)
            reviewed_dir = self.review_and_edit_labels(split=split)

        yaml_path = self.create_dataset_yaml(reviewed_dir)

        models_dir = self.output_dir / 'models'
        best_model_path = models_dir / 'human_detector' / 'weights' / 'best.pt'

        if best_model_path.exists():
            print('\n' + '-' * 60)
            print('Step 5: Using Existing Trained Model')
            print('-' * 60)
            print(f'Found existing trained model at: {best_model_path}')
            print('Skipping training and using existing weights.')
            train_metrics = {
                'final_epoch': 'N/A (using existing model)',
                'box_loss': 'N/A',
                'cls_loss': 'N/A',
                'dfl_loss': 'N/A',
            }
        else:
            print('\n' + '-' * 60)
            print('Step 5: Training Student Model')
            print('-' * 60)
            print('No existing model found. Starting training...')
            best_model_path, train_metrics = self.train_student(
                yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
            )

        val_metrics, test_metrics = self.evaluate_student(best_model_path, yaml_path) # evaluate student model

        predictions_dir = self.visualize_predictions(best_model_path, reviewed_dir, splits=['val', 'test']) # visualize predictions

        export_path = self.export_model(best_model_path, format='torchscript') # export model

        print('\n' + '-' * 60)
        print('FINAL SUMMARY')
        print('-' * 60)

        print('\nDataset Statistics:')
        for split in ['train', 'val', 'test']:
            s = label_stats[split]
            print(f'  {split.capitalize()}:')
            print(f'    Total: {s["total"]} images')
            print(f'    With humans: {s["with_humans"]} ({s["percent_with_humans"]:.1f}%)')
            print(f'    Total detections: {s["total_humans"]}')

        print('\nModel Performance:')
        print(f'  Validation Accuracy (mAP@0.5): {val_metrics["mAP50"]:.4f}')
        print(f'  Test Accuracy (mAP@0.5): {test_metrics["mAP50"]:.4f}')
        print(f'  Test Precision: {test_metrics["precision"]:.4f}')
        print(f'  Test Recall: {test_metrics["recall"]:.4f}')

        print('\nOutput Locations:')
        print(f'  Split images: {split_dir}')
        print(f'  Original labels (YOLO format): {model_readable_dir}')
        print(f'  Reviewed labels (YOLO format): {reviewed_dir}')
        print(f'  Human-readable (annotated images): {self.output_dir / "human_readable"}')
        print(f'  Comparison (side-by-side): {self.output_dir / "comparison"}')
        print(f'  Model predictions (val/test): {predictions_dir}')
        print(f'  Best model: {best_model_path}')
        print(f'  Exported model: {export_path}')

        print('\n' + '-' * 60)
        print('Pipeline complete!')
        print('-' * 60)


def main():
    """
    main execution
    """
    import argparse

    parser = argparse.ArgumentParser(description='YOLO Human Detection Knowledge Distillation')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='input',
        help='Directory containing input PNG images (default: input/)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='modeling',
        help='Output directory (default: modeling/)',
    )
    parser.add_argument(
        '--teacher-model',
        type=str,
        default='rtdetr-x.pt',
        help=(
            'Teacher model: rtdetr-x.pt, rtdetr-l.pt, yolov9e.pt, yolov10x.pt, '
            'yolov8x.pt, detectron2, groundingdino (default: rtdetr-x.pt)'
        ),
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs (default: 100)',
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size (default: 640)',
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16)',
    )
    parser.add_argument(
        '--teacher-conf',
        type=float,
        default=0.25,
        help='Teacher confidence threshold (default: 0.25, lowered for better detection)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        help='Device: mps, cuda, or cpu (default: mps for MacBook)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    distiller = YOLOHumanDistillation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        teacher_model=args.teacher_model,
    )
    distiller.teacher_conf_threshold = args.teacher_conf

    distiller.run_full_pipeline(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == '__main__':
    main()
