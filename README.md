## YOLO Picar-X Project

YOLO-based human detection setup for Picar-X with model distillation.

### Files

- `client_yolo.py` - client (Picar-X/edge) sends frames, runs inference
- `server_yolo.py` - server receives data, runs YOLO, sends results
- `model_training/deploy_inference.py` - standalone inference on images/video
- `model_training/yolo_distillation.py` - distillation/training code
- `best.pt` - current best model checkpoint

### Quick start

**Server**:
```bash
python server_yolo.py
```

**Client** (on Picar-X):
```bash
python client_yolo.py
```

**Local inference**:
```bash
python model_training/deploy_inference.py
```

Adjust paths, camera index, and IP/ports in scripts for your setup.

### Training

Full dataset not included. To retrain:
- Organize data in YOLO format (train/val images + labels)
- Point `yolo_distillation.py` at your data directories
- Run with your preferred settings

