# VILA-U MJHQ-30K Evaluation Pipeline

This evaluation pipeline provides FID and CLIP Score evaluation for VILA-U image generation models using the MJHQ-30K benchmark dataset.

## Overview

The pipeline is adapted from the HART MJHQ-30K evaluation implementation and includes:
- **Image Generation**: Generate images using VILA-U models for MJHQ-30K prompts
- **FID Score**: Measure image quality using Fréchet Inception Distance
- **CLIP Score**: Evaluate text-image alignment using CLIP models
- **Category Filtering**: Support for evaluating specific categories (people, animals, objects, etc.)
- **Results Tracking**: Optional integration with Weights & Biases

## Quick Start

### 1. Setup Dependencies

```bash
# Install required dependencies
pip install torch torchvision numpy tqdm pillow
pip install clip-by-openai cleanfid

# Install VILA-U (if not already installed)
# Follow VILA-U installation instructions
```

### 2. Download MJHQ-30K Dataset

Download the MJHQ-30K dataset from Hugging Face:

```bash
# Visit: https://huggingface.co/datasets/playgroundai/MJHQ-30K
# Download:
# - meta_data.json (contains prompts and metadata)
# - mjhq30k_imgs/ (contains reference images organized by category)
```

### 3. Run Evaluation

#### Full Evaluation (Generate + FID + CLIP)
```bash
./run_mjhq_evaluation.sh \
    --model-path /path/to/vila-u/model \
    --mjhq-metadata /path/to/MJHQ30K/meta_data.json \
    --mjhq-images /path/to/MJHQ30K/mjhq30k_imgs
```

#### Generate Images Only
```bash
./run_mjhq_evaluation.sh --generate-only \
    --model-path /path/to/vila-u/model \
    --mjhq-metadata /path/to/meta_data.json
```

#### Evaluate Existing Images
```bash
./run_mjhq_evaluation.sh --evaluate-only \
    --mjhq-metadata /path/to/meta_data.json \
    --mjhq-images /path/to/mjhq30k_imgs
```

## Usage Examples

### Basic Evaluation
```bash
# Full MJHQ-30K evaluation on "people" category
./run_mjhq_evaluation.sh \
    --model-path /models/vila-u-7b \
    --mjhq-metadata /data/MJHQ30K/meta_data.json \
    --mjhq-images /data/MJHQ30K/mjhq30k_imgs \
    --category-filter people \
    --output-dir ./results
```

### Batch Generation
```bash
# Use batch generation for faster processing
./run_mjhq_evaluation.sh \
    --model-path /models/vila-u-7b \
    --mjhq-metadata /data/meta_data.json \
    --mjhq-images /data/mjhq30k_imgs \
    --batch-size 8  # Process 8 prompts per batch

# Sequential generation (batch-size 1)
./run_mjhq_evaluation.sh \
    --model-path /models/vila-u-7b \
    --mjhq-metadata /data/meta_data.json \
    --mjhq-images /data/mjhq30k_imgs \
    --batch-size 1  # One prompt at a time
```

### Quick Testing
```bash
# Test with limited samples
./run_mjhq_evaluation.sh \
    --model-path /models/vila-u-7b \
    --mjhq-metadata /data/meta_data.json \
    --mjhq-images /data/mjhq30k_imgs \
    --max-samples 100 \
    --category-filter people
```

### FID Only
```bash
# Compute only FID score
./run_mjhq_evaluation.sh --fid-only \
    --model-path /models/vila-u-7b \
    --mjhq-metadata /data/meta_data.json \
    --mjhq-images /data/mjhq30k_imgs
```

### CLIP Only
```bash
# Compute only CLIP score
./run_mjhq_evaluation.sh --clip-only \
    --model-path /models/vila-u-7b \
    --mjhq-metadata /data/meta_data.json
```

### Direct Evaluations
```bash
# Direct FID between two directories
./run_mjhq_evaluation.sh --direct-fid \
    /path/to/real/images \
    /path/to/generated/images

# Direct CLIP score computation
./run_mjhq_evaluation.sh --direct-clip \
    /path/to/generated/images \
    /path/to/metadata.json
```

### Custom Configuration
```bash
# Custom evaluation settings
python compute_fid_clip_mjhq.py \
    --model_path /models/vila-u-7b \
    --mjhq_metadata_path /data/meta_data.json \
    --mjhq_images_path /data/mjhq30k_imgs \
    --output_dir ./results \
    --cfg_scale 3.5 \
    --generation_nums 2 \
    --category_filter people \
    --max_samples 1000 \
    --clip_model ViT-B/32
```

## Configuration Options

### Model Settings
- `--model-path`: Path to VILA-U model directory
- `--cfg`: CFG scale for image generation (default: 3.0)
- `--generation-nums`: Number of images per prompt (default: 1)
- `--seed`: Random seed for reproducibility

### Dataset Settings  
- `--mjhq-metadata`: Path to MJHQ-30K metadata.json file
- `--mjhq-images`: Path to MJHQ-30K images directory
- `--category-filter`: Category to evaluate (people, animals, objects, etc.)
- `--max-samples`: Limit number of samples for testing

### Evaluation Settings
- `--clip-model`: CLIP model to use (default: ViT-L/14)
- `--device`: Device for computation (default: cuda)
- `--batch-size`: Batch size for generation (default: 4)
  - `1`: Sequential generation (one prompt at a time)
  - `>1`: Batch generation (multiple prompts per batch for faster processing)

### Output Settings
- `--output-dir`: Directory for results and generated images
- `--exp-name`: Experiment name for results files

### Modes
- `--generate-only`: Only generate images, skip evaluation
- `--evaluate-only`: Only evaluate existing images
- `--fid-only`: Only compute FID score
- `--clip-only`: Only compute CLIP score

## File Structure

```
vila-u/evaluation/
├── compute_fid_clip_mjhq.py      # Main evaluation script
├── run_mjhq_evaluation.sh        # Convenient shell script
├── utils.py                      # Utility functions for tracking
├── README_MJHQ.md               # This file
└── ...
```

## Output Structure

```
results/
├── generated/                    # Generated images organized by category
│   ├── people/
│   │   ├── 1234.png
│   │   ├── 5678.png
│   │   └── ...
│   ├── animals/
│   └── ...
└── mjhq_results.json            # Evaluation results
```

### Results File Format
```json
{
  "fid_score": 15.23,
  "clip_score": 0.285,
  "clip_score_std": 0.042,
  "clip_samples": 2500,
  "num_samples": 2500,
  "config": {
    "model_path": "/path/to/model",
    "category_filter": "people",
    "cfg_scale": 3.0,
    "seed": 42,
    "clip_model": "ViT-L/14"
  }
}
```

## MJHQ-30K Dataset Structure

The MJHQ-30K dataset should be organized as:
```
MJHQ30K/
├── meta_data.json               # Metadata with prompts and categories
└── mjhq30k_imgs/               # Reference images
    ├── people/
    ├── animals/
    ├── objects/
    └── ...
```

### Metadata Format
```json
{
  "1234": {
    "prompt": "A photo of a person wearing a blue shirt",
    "category": ["people"],
    "width": 1024,
    "height": 1024
  }
}
```

## Evaluation Metrics

### FID Score (Fréchet Inception Distance)
Measures the quality of generated images by comparing feature distributions:
- **< 10**: Excellent quality, very similar to real images
- **10-20**: Good quality, minor differences from real images
- **20-50**: Moderate quality, noticeable differences
- **> 50**: Poor quality, significant differences

### CLIP Score
Evaluates text-image alignment using CLIP embeddings:
- **> 0.3**: Excellent text-image alignment
- **0.25-0.3**: Good alignment with minor issues
- **0.2-0.25**: Moderate alignment
- **< 0.2**: Poor text-image correspondence

## Category Filters

Supported MJHQ-30K categories:
- **people**: Human subjects
- **animals**: Animal subjects
- **objects**: Inanimate objects
- **food**: Food items
- **landscapes**: Natural scenery
- **architecture**: Buildings and structures

## Troubleshooting

### Common Issues

1. **cleanfid not found**
   ```bash
   pip install cleanfid
   ```

2. **CLIP model not found**
   ```bash
   pip install clip-by-openai
   ```

3. **VILA-U model loading fails**
   - Check model path exists and is accessible
   - Verify VILA-U is properly installed
   - Ensure model format is compatible

4. **MJHQ dataset not found**
   - Download from: https://huggingface.co/datasets/playgroundai/MJHQ-30K
   - Verify paths in script configuration
   - Check file permissions

5. **CUDA out of memory**
   - Reduce `--batch-size`
   - Use `--max-samples` for testing
   - Process categories separately

### Memory Issues

For large evaluations:
- Reduce batch size to 1-2
- Use `--max-samples` for testing
- Process one category at a time
- Run generation and evaluation separately:
  ```bash
  # Generate first
  ./run_mjhq_evaluation.sh --generate-only --model-path /path/to/model
  
  # Evaluate later  
  ./run_mjhq_evaluation.sh --evaluate-only
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
- **Category Support**: Full category filtering support
- **Pipeline Structure**: Maintains same evaluation logic but adapted for VILA-U

## Performance Benchmarks

Typical performance on MJHQ-30K "people" category (2500 samples):

### Sequential Generation (batch-size=1)
- **Generation Time**: ~2-3 hours (depending on model size and hardware)
- **FID Computation**: ~5-10 minutes
- **CLIP Computation**: ~10-15 minutes
- **Total Evaluation**: ~3-4 hours for full pipeline

### Batch Generation (batch-size=4-8)
- **Generation Time**: ~1-1.5 hours (30-50% faster than sequential)
- **FID Computation**: ~5-10 minutes
- **CLIP Computation**: ~10-15 minutes
- **Total Evaluation**: ~1.5-2 hours for full pipeline

**Note**: Batch generation provides significant speedup but requires more GPU memory. Optimal batch size depends on your GPU memory capacity and model size.

## Contributing

To extend the evaluation pipeline:
1. Modify `compute_fid_clip_mjhq.py` for new evaluation metrics
2. Update `run_mjhq_evaluation.sh` for new command-line options
3. Add new category filters or CLIP models as needed

## References

- [MJHQ-30K Dataset](https://huggingface.co/datasets/playgroundai/MJHQ-30K)
- [FID Paper](https://arxiv.org/abs/1706.08500)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [CleanFID Implementation](https://github.com/GaParmar/clean-fid)
- [HART Evaluation Implementation](../hart/evaluation/)