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

### Car & outcome
I got this car: https://www.digikey.com/en/products/detail/sunfounder/CN0351D/16612467?gclsrc=aw.ds&gad_source=1&gad_campaignid=20837516636&gbraid=0AAAAADrbLlg0RH3SusqKgtxdiH-j5BoEm

Here is the final video of my build and results: https://drive.google.com/file/d/1oVAqmW7OkOZMxO86Y-QGLEQdU6DEHm48/view?usp=sharing

Here is the final report: https://drive.google.com/file/d/1VKXC3zpU5ffOHSuDS3hOng7SVNHEv9l3/view?usp=sharing
