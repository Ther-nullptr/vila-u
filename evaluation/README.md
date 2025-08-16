# VILA-U Image Generation Evaluation

This repository provides comprehensive evaluation pipelines for VILA-U image generation models using multiple benchmarks:

## Available Evaluations

### 1. GenEval - Compositional Reasoning
Evaluates compositional understanding using object detection, counting, and spatial relationships.
- **Script**: `run_geneval_evaluation.sh`
- **Guide**: This README
- **Metrics**: Object presence, counting accuracy, spatial relations

### 2. MJHQ-30K - FID and CLIP Scores  
Evaluates image quality and text-image alignment using the MJHQ-30K benchmark.
- **Script**: `run_mjhq_evaluation.sh`
- **Guide**: [README_MJHQ.md](README_MJHQ.md)
- **Metrics**: FID Score, CLIP Score

---

# GenEval Compositional Evaluation

This pipeline provides compositional reasoning evaluation for VILA-U image generation models using the GenEval benchmark.

## Overview

The pipeline is adapted from the HART evaluation implementation and includes:
- **Image Generation**: Generate images using VILA-U models
- **Object Detection**: Detect objects using MMDetection models  
- **Compositional Evaluation**: Evaluate object presence, counting, and spatial relationships
- **Results Tracking**: Optional integration with Weights & Biases

## Quick Start

### 1. Setup Dependencies

```bash
# Install required dependencies
pip install opencv-python numpy torch tqdm pillow
pip install mmdet  # For object detection evaluation

# Install VILA-U (if not already installed)
# Follow VILA-U installation instructions
```

### 2. Download GenEval Dataset

Download the GenEval prompts file and save as `geneval_prompts.jsonl`:

```bash
# Download from GenEval repository (format the data as JSONL)
# Each line should be: {"tag": "category", "include": [{"class": "object", "count": N}], "prompt": "text"}

# For testing, you can use the provided example:
cp geneval_prompts_example.jsonl geneval_prompts.jsonl
```

### 3. Run Evaluation

#### Full Evaluation (Generate + Evaluate)
```bash
./run_geneval_evaluation.sh --model-path /path/to/vila-u/model
```

#### Generate Images Only
```bash
./run_geneval_evaluation.sh --generate-only --model-path /path/to/vila-u/model
```

#### Evaluate Existing Images
```bash
./run_geneval_evaluation.sh --evaluate-only /path/to/generated/images
```

## Usage Examples

### Basic Evaluation
```bash
# Run full GenEval evaluation
./run_geneval_evaluation.sh \
    --model-path /models/vila-u-7b \
    --output-dir ./results \
    --cfg 3.0 \
    --generation-nums 1
```

### Quick Testing
```bash
# Test with limited prompts
./run_geneval_evaluation.sh \
    --model-path /models/vila-u-7b \
    --max-prompts 100 \
    --output-dir ./test_results
```

### Custom Configuration
```bash
# Custom evaluation settings
python evaluate_geneval.py \
    --model_path /models/vila-u-7b \
    --prompt_file geneval_prompts.jsonl \
    --detector_model_path ./checkpoints/mask2former.pth \
    --detector_config_path "" \
    --output_dir ./results \
    --cfg_scale 3.0 \
    --generation_nums 2 \
    --conf_threshold 0.3
```

## Configuration Options

### Model Settings
- `--model-path`: Path to VILA-U model directory
- `--cfg`: CFG scale for image generation (default: 3.0)
- `--generation-nums`: Number of images per prompt (default: 1)

### Evaluation Settings  
- `--detector-checkpoint`: Path to MMDetection model weights
- `--conf-threshold`: Detection confidence threshold (default: 0.3)
- `--max-prompts`: Limit number of prompts for testing

### Output Settings
- `--output-dir`: Directory for results and generated images
- `--exp-name`: Experiment name for results files

### Tracking (Optional)
- `--report-to wandb`: Enable Weights & Biases tracking
- `--tracker-project-name`: W&B project name

## File Structure

```
vila-u/evaluation/
├── evaluate_geneval.py           # GenEval compositional evaluation
├── compute_fid_clip_mjhq.py      # MJHQ-30K FID and CLIP evaluation
├── utils.py                      # Utility functions for tracking
├── run_geneval_evaluation.sh     # GenEval evaluation script
├── run_mjhq_evaluation.sh        # MJHQ-30K evaluation script
├── demo_folder_structure.py      # Demo script for folder structure
├── test_pipeline.py              # GenEval pipeline testing
├── test_mjhq_pipeline.py         # MJHQ-30K pipeline testing
├── README.md                     # This file (GenEval evaluation)
├── README_MJHQ.md               # MJHQ-30K evaluation guide
├── geneval_prompts_example.jsonl # Example GenEval data
└── checkpoints/                  # Auto-downloaded detection models
    └── mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
```

## Output Structure

```
results/
├── generated_images/             # Generated images in structured folders
│   ├── 00000/                   # First prompt
│   │   ├── metadata.jsonl       # Original prompt metadata
│   │   └── samples/
│   │       ├── 0000.png         # First generated sample
│   │       ├── 0001.png         # Second generated sample (if multiple)
│   │       └── ...
│   ├── 00001/                   # Second prompt
│   │   ├── metadata.jsonl
│   │   └── samples/
│   │       └── 0000.png
│   └── ...
└── vila_u_geneval_evaluation_geneval_results.json  # Evaluation results
```

### Folder Structure Details
- Each prompt gets a folder named with 5-digit zero-padded index (00000, 00001, etc.)
- `metadata.jsonl` contains the original prompt data from the input file
- `samples/` folder contains generated images named as 4-digit zero-padded indices
- `grid.png` is optional and not generated by default

### Demo Folder Structure
You can preview the expected folder structure using:
```bash
python demo_folder_structure.py
# Creates demo_output/ with sample structure

python demo_folder_structure.py --validate-only --output-dir /path/to/existing/folder
# Validates existing folder structure
```

## GenEval Data Format

The evaluation uses JSONL format where each line contains:
```json
{"tag": "single_object", "include": [{"class": "cow", "count": 1}], "prompt": "a photo of a cow"}
```

### Supported Tags
- **single_object**: Single object recognition
- **two_object**: Two object co-occurrence  
- **counting**: Numerical counting (e.g., "three apples")
- **colors**: Color attribute recognition
- **position**: Spatial relationship understanding
- **color_attr**: Multi-object color attributes

## Evaluation Metrics

The pipeline evaluates based on tag categories:

1. **Single Object**: Basic object recognition (≥50% threshold)
2. **Two Object**: Both objects must be present (≥90% threshold)
3. **Counting**: Exact count accuracy (≥80% threshold)
4. **Colors**: Object presence with color context (≥70% threshold)
5. **Position**: Spatial relationships (≥60% threshold)
6. **Color Attr**: Multi-object color binding (≥70% threshold)

### Score Interpretation
- **> 0.8**: Excellent compositional understanding
- **0.6-0.8**: Good performance with room for improvement  
- **0.4-0.6**: Moderate performance, significant issues
- **< 0.4**: Poor compositional understanding

## Detection Model

The pipeline uses Mask2Former for object detection:
- **Model**: `mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco`
- **Auto-download**: Checkpoint downloaded automatically if not present
- **Custom models**: Can specify custom detection checkpoints

## Troubleshooting

### Common Issues

1. **MMDetection not found**
   ```bash
   pip install mmdet
   ```

2. **Detection checkpoint download fails**
   ```bash
   # Manual download
   mkdir -p checkpoints
   wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220326_224521-11a44721.pth \
        -O checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
   ```

3. **VILA-U model loading fails**
   - Check model path exists and is accessible
   - Verify VILA-U is properly installed
   - Ensure model format is compatible

4. **GenEval prompts not found**
   ```bash
   # Use example file for testing
   cp geneval_prompts_example.jsonl geneval_prompts.jsonl
   
   # Or format your own GenEval data as JSONL
   ```

### Memory Issues

For large evaluations:
- Reduce `--generation-nums` to 1
- Use `--max-prompts` for testing
- Run generation and evaluation separately:
  ```bash
  # Generate first
  ./run_geneval_evaluation.sh --generate-only --model-path /path/to/model
  
  # Evaluate later  
  ./run_geneval_evaluation.sh --evaluate-only ./results/generated_images
  ```

## Integration with VILA-U

The evaluation script integrates with VILA-U through:

```python
import vila_u

# Load model
model = vila_u.load(model_path)

# Generate images
response = model.generate_image_content(prompt, cfg_scale, generation_nums)
```

Ensure your VILA-U installation supports the `generate_image_content` method.

## Comparison with HART

Key differences from HART evaluation:
- **Model Interface**: Uses VILA-U's `generate_image_content` instead of HART's interface
- **Generation Parameters**: Adapted CFG scale defaults (3.0 vs 4.5)
- **Image Saving**: Modified for VILA-U's output format
- **Pipeline Structure**: Maintains same evaluation logic but adapted for VILA-U

## Contributing

To extend the evaluation pipeline:
1. Modify `evaluate_geneval.py` for new evaluation metrics
2. Update `run_geneval_evaluation.sh` for new command-line options
3. Add new detection models in the checkpoint handling section

## References

- [GenEval Paper](https://arxiv.org/abs/2402.06292)
- [GenEval Repository](https://github.com/djghosh13/geneval)
- [HART Evaluation Implementation](../hart/evaluation/)
- [MMDetection](https://github.com/open-mmlab/mmdetection)