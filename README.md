# Adversarial Patch Defense Project (CS682: Neural Networks)

## Overview
This project implements atargeted  adversarial patch attack based on [1] and defense mechanism using a pretrained ResNet34 model on the Tiny ImageNet dataset. The adversarial patches are designed to mislead the classifier into predicting a target class, and a defense mechanism (Segment and Complete [2]) is applied to mitigate the attack. The code evaluates the effectiveness of both the attack and defense across different patch sizes and positions.

## Setup Instructions
1. **Install Dependencies**:
   ```bash
   pip install torch torchvision pytorch-lightning tqdm numpy matplotlib seaborn kornia tabulate
   ```

2. **Prepare Dataset and Models**:
   - The `main.py` script automatically downloads the Tiny ImageNet dataset and pretrained patches to the `data/` and `saved_models/` directories.
   - Ensure the `ckpts/` directory contains the pretrained weights for the patch detectors (`coco_at.pth` and `apricot_mask.pth`).
   - If not present, update the paths in `defense_utils.py` or obtain the weights as specified in the original paper.

3. **Directory Setup**:
   - Create `data/`, `saved_models/`, and `ckpts/` directories if they do not exist.
   - Ensure write permissions for downloading and unzipping files.

## Usage
Run the main script to execute the entire pipeline:
python main.py

## Results
For results and the detailed overview of the project, check the project report - cs682_project_report_final.

## References
[1] https://arxiv.org/abs/1712.09665

[2] https://arxiv.org/abs/2112.04532
