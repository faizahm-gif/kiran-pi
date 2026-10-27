# Banana leaf deficiency classifier — Raspberry Pi inference

This folder is everything needed to classify one banana-leaf photo. Training code is not included.

The checkpoint is `checkpoints/best` from the training run (`lmfoaa/checkpoints/best`):

- Backbone: DINOv3 ViT-B/16
- Epoch 10 of fine-tuning
- Validation macro F1: 0.843
- Validation accuracy: 0.842

Classes: Boron, Calcium, Healthy, Iron, Magnesium, Manganese, Potassium, Sulphur, Zinc.

`model.pt` is about 331 MB. Copy this whole `pi_inference` folder to the Pi. The weight file alone is not enough; `processor/` and `backbone_config/` must stay beside it.

Use a 64-bit Raspberry Pi OS and a Pi with at least 4 GB of RAM (Pi 5 is more comfortable). The first prediction loads the full network into memory and can take one to two minutes. Later photos in the same process are faster; each new `python inference.py` call loads the weights again.

## 1. Copy the folder

From the PC, copy `pi_inference` onto the Pi (USB, `scp`, or a shared drive). Example from the PC:

```powershell
scp -r D:\new_dataset\split\pi_inference pi@raspberrypi.local:~/pi_inference
```

## 2. Install

On the Pi, in a terminal:

```bash
sudo apt update
sudo apt install -y rpicam-apps python3-picamera2
cd ~/pi_inference
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`--system-site-packages` lets the virtual environment see `picamera2` from apt. If that import is missing, capture still works through the `rpicam-still` command.

If the PyTorch CPU index has no matching ARM wheel, install the Pi wheel instead, then the rest of the requirements:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

Check that PyTorch imports on the Pi:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False` for CUDA is expected. Inference runs on the CPU.

## 3. Run from the Pi Camera

Connect the camera, then confirm the Pi sees it:

```bash
rpicam-hello --list-cameras
```

Classify a leaf by pointing the camera at it:

```bash
cd ~/pi_inference
source .venv/bin/activate
python inference.py --device cpu --segment-mode classic
```

The camera opens, waits for you to frame the leaf, and captures when you press Enter. Over SSH with no preview window, add `--no-preview`. The photo is saved under `outputs/captures/leaf_<date>_<time>.jpg`, then classified.

```bash
python inference.py --device cpu --segment-mode classic --no-preview
```

To classify a photo you already have, pass the file and the camera stays closed:

```bash
python inference.py --image /home/pi/leaf.jpg --device cpu --segment-mode classic
```

Printed output is the predicted deficiency, its confidence, and the full class ranking. Files are written to `outputs/inference/<image name>/`:

- `prediction.json` — class, confidence, and every class probability
- `segmented_leaf.png` — leaf after background removal
- `overlay.jpg` — original with the leaf outline next to the cut-out
- `confidence.png` — bar chart (skipped if you pass `--no-chart`)

### Which segmentation flag to use

| Photo | Command |
| --- | --- |
| Leaf on a black background (same style as the training set) | `python inference.py --image leaf.jpg --device cpu` |
| Soil, sky, hands, or a bench, and you want a light install | add `--segment-mode classic` |
| Leaf is already cut out on black | add `--no-segment` only if you do not want another crop |

`--segment-mode classic` uses the black-background mask, then GrabCut. It does not download SAM 2. That is the right default on a Pi.

`--segment-mode auto` does the same for black-background photos. For a real background it tries SAM 2, then falls back to GrabCut when SAM is not installed.

## 4. Optional SAM 2

SAM 2 is slower and uses more RAM. Skip it unless GrabCut is cutting the leaf poorly.

```bash
pip install ultralytics
python inference.py --image leaf.jpg --device cpu --segment-mode sam --sam-weights sam2.1_b.pt
```

The first SAM run downloads `sam2.1_b.pt` (about 150 MB) into the current folder.

## Notes

- No Hugging Face login is required. The backbone is rebuilt from `checkpoints/best/backbone_config` and the trained weights in `model.pt`.
- Do not separate `model.pt` from `meta.json`, `processor/`, and `backbone_config/`.
- If the Pi runs out of memory, close other programs, use `--segment-mode classic`, and avoid `--segment-mode sam`.
