'''deployment inference script for human detection model
uses argmax with confidence threshold > 0.45

this is used onboard the pi car to detect humans 
using the pretrained model from laptop (yolo_distillation.py)

- model_path: path to trained model weights
- image_path: path to input image
- conf_threshold: optional override for confidence threshold'''

import sys
from pathlib import Path
from yolo_distillation import YOLOHumanDistillation

def main():
    # check args count
    if len(sys.argv) < 3:
        print('Usage: python deploy_inference.py <model_path> <image_path> [conf_threshold]')
        print('Example: python deploy_inference.py modeling/models/human_detector/weights/best.pt input/test.png 0.45')
        sys.exit(1)
    
    # grab input args
    model_path = Path(sys.argv[1])
    image_path = Path(sys.argv[2])
    conf_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45
    
    # check paths
    if not model_path.exists():
        print(f'Error: Model not found at {model_path}')
        sys.exit(1)
    
    if not image_path.exists():
        print(f'Error: Image not found at {image_path}')
        sys.exit(1)
    
    # run deployment inference
    result = YOLOHumanDistillation.detect_human_deployed(
        model_path=model_path,
        image_path_or_array=str(image_path),
        conf_threshold=conf_threshold
    )
    
    # print results
    if result:
        print('Human detected!')
        print(f'  Confidence: {result["confidence"]:.4f}')
        print(f'  Bounding box (normalized): {result["bbox"]}')
        print(f'  Bounding box (absolute): {result["bbox_abs"]}')
        return result
    
    print('No human detected (confidence below threshold or no detections)')
    return None

if __name__ == '__main__':
    main()
