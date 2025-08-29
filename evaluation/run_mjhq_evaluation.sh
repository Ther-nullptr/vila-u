#!/bin/bash

# VILA-U Model MJHQ-30K FID and CLIP Evaluation Script
# Adapted from HART evaluation for VILA-U model

set -e  # Exit on any error

# =============================================================================
# Configuration - Update these paths for your setup
# =============================================================================

# Model paths (REQUIRED for generation mode)
MODEL_PATH="/home/wyj24/models/vila-u-7b-256/"  # Update this path

# Output directory
OUTPUT_DIR="./mjhq_evaluation_results"
DEVICE="cuda"

# Dataset paths for MJHQ-30K evaluation
MJHQ_METADATA="/data/MJHQ-30K/meta_data.json"  # Update this path
MJHQ_IMAGES="/data/MJHQ-30K/imgs"      # Update this path

# Generation settings
CFG_SCALE=3.0
SEED=42
GENERATION_NUMS=1

# Evaluation settings
CATEGORY_FILTER="food"  # Set to "" for all categories, or specify: people, animals, objects, etc.
MAX_SAMPLES=""            # Set to limit samples for testing, empty for all
CLIP_MODEL="ViT-L/14"

# Technical settings
BATCH_SIZE=4  # Batch size for image generation (1 for sequential, >1 for batch)

# Experiment tracking (optional)
EXP_NAME="vila_u_mjhq_evaluation"
REPORT_TO="none"  # Set to "wandb" to enable tracking
TRACKER_PROJECT="vila-u-evaluation"

# =============================================================================
# Script execution - Don't modify below unless you know what you're doing
# =============================================================================

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"

echo "=== VILA-U Model MJHQ-30K Evaluation ==="
echo "Evaluating image quality using FID and CLIP scores on MJHQ-30K dataset"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Dataset: MJHQ-30K"
echo ""

# Utility functions
check_file() {
    if [ ! -f "$1" ]; then
        echo "Error: File not found: $1"
        return 1
    fi
    return 0
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "Error: Directory not found: $1"
        return 1
    fi
    return 0
}

print_section() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

# Parse command line arguments
DIRECT_FID=false
DIRECT_CLIP=false
REAL_PATH=""
GEN_PATH=""
METADATA_PATH=""
GENERATE_ONLY=true
EVALUATE_ONLY=true
FID_ONLY=false
CLIP_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --direct-fid)
            DIRECT_FID=true
            REAL_PATH="$2"
            GEN_PATH="$3"
            shift 3
            ;;
        --direct-clip)
            DIRECT_CLIP=true
            GEN_PATH="$2"
            METADATA_PATH="$3"
            shift 3
            ;;
        --generate-only)
            GENERATE_ONLY=true
            shift
            ;;
        --evaluate-only)
            EVALUATE_ONLY=true
            shift
            ;;
        --fid-only)
            FID_ONLY=true
            shift
            ;;
        --clip-only)
            CLIP_ONLY=true
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --mjhq-metadata)
            MJHQ_METADATA="$2"
            shift 2
            ;;
        --mjhq-images)
            MJHQ_IMAGES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --category-filter)
            CATEGORY_FILTER="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --cfg)
            CFG_SCALE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --clip-model)
            CLIP_MODEL="$2"
            shift 2
            ;;
        --generation-nums)
            GENERATION_NUMS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            echo "VILA-U MJHQ-30K Evaluation Script"
            echo ""
            echo "Usage:"
            echo "  ./run_mjhq_evaluation.sh [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  Default:      Generate images and compute FID+CLIP on MJHQ-30K"
            echo "  Generate-only: ./run_mjhq_evaluation.sh --generate-only"
            echo "  Evaluate-only: ./run_mjhq_evaluation.sh --evaluate-only"
            echo "  FID-only:     ./run_mjhq_evaluation.sh --fid-only"
            echo "  CLIP-only:    ./run_mjhq_evaluation.sh --clip-only"
            echo "  Direct FID:   ./run_mjhq_evaluation.sh --direct-fid /path/to/real /path/to/generated"
            echo "  Direct CLIP:  ./run_mjhq_evaluation.sh --direct-clip /path/to/generated /path/to/metadata.json"
            echo ""
            echo "Options:"
            echo "  --model-path PATH         Path to VILA-U model directory"
            echo "  --mjhq-metadata PATH      Path to MJHQ-30K metadata.json"
            echo "  --mjhq-images PATH        Path to MJHQ-30K images directory"
            echo "  --output-dir PATH         Output directory (default: ./mjhq_evaluation_results)"
            echo "  --category-filter CAT     Filter to specific category (e.g., people)"
            echo "  --max-samples N           Limit number of samples"
            echo "  --cfg SCALE               CFG scale (default: 3.0)"
            echo "  --seed N                  Random seed (default: 42)"
            echo "  --device DEVICE           Device (default: cuda)"
            echo "  --clip-model MODEL        CLIP model (default: ViT-L/14)"
            echo "  --generation-nums N       Number of images per prompt (default: 1)"
            echo "  --batch-size N            Batch size for generation (default: 4)"
            echo "  --generate-only           Only generate images"
            echo "  --evaluate-only           Only evaluate existing images"
            echo "  --fid-only                Only compute FID score"
            echo "  --clip-only               Only compute CLIP score"
            echo "  --help, -h                Show this help"
            echo ""
            echo "Examples:"
            echo "  # Full MJHQ-30K evaluation"
            echo "  ./run_mjhq_evaluation.sh --model-path /path/to/model --mjhq-metadata /path/to/meta.json --mjhq-images /path/to/images"
            echo ""
            echo "  # Quick test with limited samples"
            echo "  ./run_mjhq_evaluation.sh --max-samples 100 --category-filter people"
            echo ""
            echo "  # Generate images only"
            echo "  ./run_mjhq_evaluation.sh --generate-only --model-path /path/to/model"
            echo ""
            echo "  # Batch generation for faster processing"
            echo "  ./run_mjhq_evaluation.sh --model-path /path/to/model --batch-size 8"
            echo ""
            echo "  # Sequential generation (batch-size 1)"
            echo "  ./run_mjhq_evaluation.sh --model-path /path/to/model --batch-size 1"
            echo ""
            echo "  # Direct FID between two directories"
            echo "  ./run_mjhq_evaluation.sh --direct-fid /path/to/real/images /path/to/generated/images"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Direct FID Mode
# =============================================================================

if [ "$DIRECT_FID" = true ]; then
    print_section "Direct FID Computation Mode"
    
    echo "Computing FID between:"
    echo "  Real images: $REAL_PATH"
    echo "  Generated images: $GEN_PATH"
    echo ""
    
    # Validate paths
    if ! check_dir "$REAL_PATH"; then
        echo "❌ ERROR: Real images directory not found: $REAL_PATH"
        exit 1
    fi
    
    if ! check_dir "$GEN_PATH"; then
        echo "❌ ERROR: Generated images directory not found: $GEN_PATH"
        exit 1
    fi
    
    echo "Running direct FID computation..."
    python -c "
from cleanfid import fid
import sys
fid_score = fid.compute_fid('$REAL_PATH', '$GEN_PATH', device='$DEVICE')
print(f'FID Score: {fid_score:.4f}')
"
    
    echo ""
    echo "✅ Direct FID computation completed!"
    exit 0
fi

# =============================================================================
# Direct CLIP Mode
# =============================================================================

if [ "$DIRECT_CLIP" = true ]; then
    print_section "Direct CLIP Score Computation Mode"
    
    echo "Computing CLIP score for:"
    echo "  Generated images: $GEN_PATH"
    echo "  Metadata file: $METADATA_PATH"
    echo ""
    
    # Validate paths
    if ! check_dir "$GEN_PATH"; then
        echo "❌ ERROR: Generated images directory not found: $GEN_PATH"
        exit 1
    fi
    
    if ! check_file "$METADATA_PATH"; then
        echo "❌ ERROR: Metadata file not found: $METADATA_PATH"
        exit 1
    fi
    
    echo "Running direct CLIP score computation..."
    python compute_fid_clip_mjhq.py \
        --model_path $MODEL_PATH \
        --mjhq_metadata_path "$METADATA_PATH" \
        --mjhq_images_path "$MJHQ_IMAGES" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --clip_model "$CLIP_MODEL" \
        --exp_name "$EXP_NAME" \
        --report_to "$REPORT_TO" \
        --tracker_project_name "$TRACKER_PROJECT" \
        --evaluate_only \
        --clip_only
    
    echo ""
    echo "✅ Direct CLIP score computation completed!"
    exit 0
fi

# =============================================================================
# Full MJHQ-30K Evaluation Mode
# =============================================================================

print_section "MJHQ-30K Evaluation Mode"

# Validate essential paths
if [ "$EVALUATE_ONLY" = false ]; then
    if [ -z "$MODEL_PATH" ] || [ "$MODEL_PATH" = "/path/to/vila-u/model" ]; then
        echo "❌ ERROR: Please specify MODEL_PATH"
        echo "Use: --model-path /path/to/your/vila-u/model"
        echo "Or update MODEL_PATH in the script configuration section"
        exit 1
    fi
    
    if ! check_dir "$MODEL_PATH"; then
        echo "❌ ERROR: VILA-U model directory not found: $MODEL_PATH"
        exit 1
    fi
fi

if [ -z "$MJHQ_METADATA" ] || [ "$MJHQ_METADATA" = "/path/to/MJHQ30K/meta_data.json" ]; then
    echo "❌ ERROR: Please specify MJHQ metadata path"
    echo "Use: --mjhq-metadata /path/to/MJHQ30K/meta_data.json"
    echo "Or update MJHQ_METADATA in the script configuration section"
    echo ""
    echo "To download MJHQ-30K dataset:"
    echo "  1. Visit: https://huggingface.co/datasets/playgroundai/MJHQ-30K"
    echo "  2. Download the dataset"
    echo "  3. Update the paths in this script"
    exit 1
fi

if [ -z "$MJHQ_IMAGES" ] || [ "$MJHQ_IMAGES" = "/path/to/MJHQ30K/mjhq30k_imgs" ]; then
    echo "❌ ERROR: Please specify MJHQ images path"
    echo "Use: --mjhq-images /path/to/MJHQ30K/mjhq30k_imgs"
    echo "Or update MJHQ_IMAGES in the script configuration section"
    exit 1
fi

echo "Validating paths..."

if ! check_file "$MJHQ_METADATA"; then
    echo "❌ ERROR: MJHQ metadata file not found: $MJHQ_METADATA"
    exit 1
fi

if ! check_dir "$MJHQ_IMAGES"; then
    echo "❌ ERROR: MJHQ images directory not found: $MJHQ_IMAGES"
    exit 1
fi

echo "✅ All paths validated successfully"
echo ""

# Display evaluation settings
echo "Evaluation Settings:"
echo "  • Category filter: ${CATEGORY_FILTER:-all}"
echo "  • Max samples: ${MAX_SAMPLES:-all}"
echo "  • CFG scale: $CFG_SCALE"
echo "  • Random seed: $SEED"
echo "  • Generation nums: $GENERATION_NUMS"
echo "  • Batch size: $BATCH_SIZE"
echo "  • CLIP model: $CLIP_MODEL"
echo ""

# Prepare evaluation arguments
EVAL_ARGS="--mjhq_metadata_path \"$MJHQ_METADATA\" \
          --mjhq_images_path \"$MJHQ_IMAGES\" \
          --output_dir \"$OUTPUT_DIR\" \
          --device $DEVICE \
          --cfg_scale $CFG_SCALE \
          --seed $SEED \
          --batch_size $BATCH_SIZE \
          --generation_nums $GENERATION_NUMS \
          --clip_model \"$CLIP_MODEL\" \
          --exp_name \"$EXP_NAME\" \
          --report_to \"$REPORT_TO\" \
          --tracker_project_name \"$TRACKER_PROJECT\""

# Add model path if not evaluate-only
if [ "$EVALUATE_ONLY" = false ]; then
    EVAL_ARGS="$EVAL_ARGS --model_path \"$MODEL_PATH\""
fi

# Add mode flags
if [ "$GENERATE_ONLY" = true ]; then
    EVAL_ARGS="$EVAL_ARGS --generate_only"
elif [ "$EVALUATE_ONLY" = true ]; then
    EVAL_ARGS="$EVAL_ARGS --evaluate_only"
fi

if [ "$FID_ONLY" = true ]; then
    EVAL_ARGS="$EVAL_ARGS --fid_only"
elif [ "$CLIP_ONLY" = true ]; then
    EVAL_ARGS="$EVAL_ARGS --clip_only"
fi

# Add optional parameters
if [ -n "$CATEGORY_FILTER" ]; then
    EVAL_ARGS="$EVAL_ARGS --category_filter \"$CATEGORY_FILTER\""
fi

if [ -n "$MAX_SAMPLES" ]; then
    EVAL_ARGS="$EVAL_ARGS --max_samples $MAX_SAMPLES"
fi

# Run evaluation
echo "Starting MJHQ-30K evaluation..."

if [ "$GENERATE_ONLY" = true ]; then
    echo "This will generate images for MJHQ-30K prompts using VILA-U"
elif [ "$EVALUATE_ONLY" = true ]; then
    echo "This will evaluate existing generated images"
else
    echo "This will:"
    echo "  1. Load VILA-U model"
    echo "  2. Generate images for MJHQ-30K prompts"
    echo "  3. Compute FID and CLIP scores"
fi
echo ""

eval "python compute_fid_clip_mjhq.py $EVAL_ARGS --model_path $MODEL_PATH"

# Display results
print_section "MJHQ Evaluation Results"

if [ "$GENERATE_ONLY" = true ]; then
    echo "✅ Image generation completed!"
    echo ""
    echo "Generated images saved to: $OUTPUT_DIR/generated/"
    echo ""
    echo "To run evaluation:"
    echo "  ./run_mjhq_evaluation.sh --evaluate-only"
    exit 0
fi

if [ -f "$OUTPUT_DIR/mjhq_results.json" ]; then
    echo "📊 MJHQ Evaluation Completed Successfully!"
    echo ""
    python -c "
import json
data = json.load(open('$OUTPUT_DIR/mjhq_results.json'))
if 'fid_score' in data:
    print(f'🎯 FID Score: {data[\"fid_score\"]:.4f}')
if 'clip_score' in data:
    print(f'🔗 CLIP Score: {data[\"clip_score\"]:.4f} ± {data.get(\"clip_score_std\", 0):.4f}')
    print(f'📊 CLIP Samples: {data.get(\"clip_samples\", 0)}')
print()
print('📋 Evaluation Details:')
print(f'  • Total Samples: {data[\"num_samples\"]}')
print(f'  • Category Filter: {data[\"config\"].get(\"category_filter\", \"all\")}')
print(f'  • CFG Scale: {data[\"config\"][\"cfg_scale\"]}')
print(f'  • Random Seed: {data[\"config\"][\"seed\"]}')
print()
print('📁 Results Location:')
print(f'  • Generated Images: $OUTPUT_DIR/generated/')
print(f'  • Results: $OUTPUT_DIR/mjhq_results.json')
"
    echo ""
    echo "✅ MJHQ evaluation completed successfully!"
else
    echo "❌ MJHQ evaluation failed - no results file found"
    echo "Check the error messages above for troubleshooting"
    exit 1
fi

# Performance guidance
echo ""
echo "📈 Score Interpretations:"
echo "  FID Score:"
echo "    • < 10:  Excellent quality, very similar to real images"
echo "    • 10-20: Good quality, minor differences from real images"
echo "    • 20-50: Moderate quality, noticeable differences"
echo "    • > 50:  Poor quality, significant differences"
echo ""
echo "  CLIP Score:"
echo "    • > 0.3:  Excellent text-image alignment"
echo "    • 0.25-0.3: Good alignment with minor issues"
echo "    • 0.2-0.25: Moderate alignment"
echo "    • < 0.2:  Poor text-image correspondence"
echo ""

echo "=== MJHQ Evaluation Complete ==="