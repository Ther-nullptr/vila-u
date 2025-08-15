#!/bin/bash

# VILA-U Model GenEval Evaluation Script
# Dedicated script for GenEval compositional reasoning evaluation

set -e  # Exit on any error

# =============================================================================
# Configuration - Update these paths for your setup
# =============================================================================

# Model paths (REQUIRED for generation mode)
MODEL_PATH="/home/wyj24/models/vila-u-7b-256/"  # Update this path

# Output directory
OUTPUT_DIR="./geneval_evaluation_results"
DEVICE="cuda"

# Detection model paths (auto-downloaded if not present)
DETECTOR_CONFIG=""  # Leave empty for default config
DETECTOR_CHECKPOINT="./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"

# GenEval settings
DOWNLOAD_GENEVAL="--download_geneval"  # Download official GenEval metadata
GENERATION_NUMS=4
MAX_PROMPTS=""  # Set to limit prompts for testing, empty for full dataset

# Generation settings
CFG_SCALE=3.0
SEED=42

# Evaluation thresholds
THRESHOLD=0.3

# Technical settings
IMG_SIZE=256
MAX_TOKEN_LENGTH=300

# Experiment tracking (optional)
EXP_NAME="vila_u_geneval_evaluation"
REPORT_TO="none"  # Set to "wandb" to enable tracking
TRACKER_PROJECT="vila-u-evaluation"

# =============================================================================
# Script execution - Don't modify below unless you know what you're doing
# =============================================================================

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"

echo "=== VILA-U Model GenEval Evaluation ==="
echo "Evaluating compositional reasoning using official GenEval dataset"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Dataset: Official GenEval"
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
EVALUATE_ONLY=false
PROMPTS_DIR=""
GENEVAL_METADATA=""
SKIP_GENERATION=false
GENERATE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --evaluate-only)
            EVALUATE_ONLY=true
            PROMPTS_DIR="$2"
            shift 2
            ;;
        --generate-only)
            GENERATE_ONLY=true
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --geneval-metadata)
            GENEVAL_METADATA="$2"
            DOWNLOAD_GENEVAL=""  # Don't download if metadata provided
            shift 2
            ;;
        --detector-checkpoint)
            DETECTOR_CHECKPOINT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-prompts)
            MAX_PROMPTS="$2"
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
        --generation-nums)
            GENERATION_NUMS="$2"
            shift 2
            ;;
        --help|-h)
            echo "VILA-U GenEval Evaluation Script"
            echo ""
            echo "Usage:"
            echo "  ./run_geneval_evaluation.sh [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  Default:      Generate images and evaluate on GenEval"
            echo "  Generate-only: ./run_geneval_evaluation.sh --generate-only"
            echo "  Evaluate-only: ./run_geneval_evaluation.sh --evaluate-only /path/to/prompts"
            echo ""
            echo "Options:"
            echo "  --model-path PATH             Path to VILA-U model directory"
            echo "  --geneval-metadata PATH       Path to GenEval metadata file"
            echo "  --detector-checkpoint PATH    Path to detection model checkpoint"
            echo "  --output-dir PATH             Output directory"
            echo "  --max-prompts N               Limit number of prompts (for testing)"
            echo "  --cfg SCALE                   CFG scale (default: 3.0)"
            echo "  --seed N                      Random seed (default: 42)"
            echo "  --device DEVICE               Device (default: cuda)"
            echo "  --generation-nums N           Number of images per prompt (default: 1)"
            echo "  --generate-only               Only generate images"
            echo "  --evaluate-only PATH          Only evaluate existing images"
            echo "  --help, -h                    Show this help"
            echo ""
            echo "Examples:"
            echo "  # Full GenEval evaluation"
            echo "  ./run_geneval_evaluation.sh --model-path /path/to/model"
            echo ""
            echo "  # Quick test with limited prompts"
            echo "  ./run_geneval_evaluation.sh --max-prompts 100"
            echo ""
            echo "  # Generate images only"
            echo "  ./run_geneval_evaluation.sh --generate-only --model-path /path/to/model"
            echo ""
            echo "  # Evaluate existing images only"
            echo "  ./run_geneval_evaluation.sh --evaluate-only /path/to/images --detector-checkpoint /path/to/checkpoint.pth"
            echo ""
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
# Evaluate-only Mode
# =============================================================================

if [ "$EVALUATE_ONLY" = true ]; then
    print_section "Evaluate-only Mode"
    
    echo "Evaluating existing images in: $PROMPTS_DIR"
    echo ""
    
    if ! check_dir "$PROMPTS_DIR"; then
        echo "❌ ERROR: Prompts directory not found: $PROMPTS_DIR"
        exit 1
    fi
    
    if [ -z "$DETECTOR_CHECKPOINT" ] || ! check_file "$DETECTOR_CHECKPOINT"; then
        echo "❌ ERROR: Detection checkpoint required for evaluation"
        echo "Please specify: --detector-checkpoint /path/to/checkpoint.pth"
        exit 1
    fi
    
    echo "Loading GenEval metadata from prompts directory..."
    
    # Find metadata file
    if [ -f "$PROMPTS_DIR/../evaluation_metadata.jsonl" ]; then
        GENEVAL_METADATA="$PROMPTS_DIR/../evaluation_metadata.jsonl"
    elif [ -f "$PROMPTS_DIR/evaluation_metadata.jsonl" ]; then
        GENEVAL_METADATA="$PROMPTS_DIR/evaluation_metadata.jsonl"
    else
        echo "❌ ERROR: Cannot find evaluation_metadata.jsonl"
        echo "Expected locations:"
        echo "  $PROMPTS_DIR/../evaluation_metadata.jsonl"
        echo "  $PROMPTS_DIR/evaluation_metadata.jsonl"
        exit 1
    fi
    
    echo "Running evaluation with existing images..."
    python evaluate_geneval.py \
        --model_path "dummy" \
        --img_path "$PROMPTS_DIR" \
        --prompt_file "$GENEVAL_METADATA" \
        --detector_model_path "$DETECTOR_CHECKPOINT" \
        --detector_config_path "" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --conf_threshold $THRESHOLD \
        --exp_name "$EXP_NAME" \
        --report_to "$REPORT_TO" \
        --tracker_project_name "$TRACKER_PROJECT" \
        --evaluate_only
    
    print_section "Evaluation Results"
    
    if [ -f "$OUTPUT_DIR/${EXP_NAME}_geneval_results.json" ]; then
        echo "🎯 GenEval Evaluation Completed Successfully!"
        echo ""
        python -c "
import json
data = json.load(open('$OUTPUT_DIR/${EXP_NAME}_geneval_results.json'))
print(f'🎯 Overall Accuracy: {data[\"overall_accuracy\"]:.3f}')
print()
print('📋 Task-specific Results:')
for key, value in data.items():
    if key.endswith('_accuracy') and key != 'overall_accuracy':
        task = key.replace('_accuracy', '').replace('_', ' ').title()
        print(f'  • {task:<15}: {value:.3f}')
"
        echo ""
        echo "✅ Evaluation completed successfully!"
    else
        echo "❌ Evaluation failed - no results file found"
        exit 1
    fi
    
    exit 0
fi

# =============================================================================
# Full GenEval Evaluation Mode
# =============================================================================

print_section "Full GenEval Evaluation Mode"

# Validate model path for generation
if [ "$EVALUATE_ONLY" = false ]; then
    if [ -z "$MODEL_PATH" ] || [ "$MODEL_PATH" = "/path/to/vila-u/model" ]; then
        echo "❌ ERROR: Please specify MODEL_PATH for image generation"
        echo "Use: --model-path /path/to/your/vila-u/model"
        echo "Or update MODEL_PATH in the script configuration section"
        exit 1
    fi
    
    if ! check_dir "$MODEL_PATH"; then
        echo "❌ ERROR: VILA-U model directory not found: $MODEL_PATH"
        exit 1
    fi
    
    echo "✅ VILA-U model path validated: $MODEL_PATH"
fi

# Check and download detection checkpoint
print_section "Detection Model Setup"

SKIP_EVALUATION_FINAL=false

if [ "$GENERATE_ONLY" = false ]; then
    if [ -n "$DETECTOR_CHECKPOINT" ] && [ "$DETECTOR_CHECKPOINT" != "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
        if ! check_file "$DETECTOR_CHECKPOINT"; then
            echo "⚠️  Detection checkpoint not found: $DETECTOR_CHECKPOINT"
            SKIP_EVALUATION_FINAL=true
        fi
    elif [ -z "$DETECTOR_CHECKPOINT" ] || [ "$DETECTOR_CHECKPOINT" = "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
        if [ ! -f "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
            echo "Detection checkpoint not found. Downloading..."
            mkdir -p checkpoints
            cd checkpoints
            
            # Download Mask2Former checkpoint
            CHECKPOINT_URL="https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220326_224521-11a44721.pth"
            
            if command -v wget >/dev/null 2>&1; then
                echo "Downloading with wget..."
                wget "$CHECKPOINT_URL" -O mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
            elif command -v curl >/dev/null 2>&1; then
                echo "Downloading with curl..."
                curl -L "$CHECKPOINT_URL" -o mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
            else
                echo "❌ Neither wget nor curl found. Please download the checkpoint manually:"
                echo "  URL: $CHECKPOINT_URL"
                echo "  Save as: ./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
                SKIP_EVALUATION_FINAL=true
            fi
            
            cd ..
            
            if [ -f "./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
                echo "✅ Checkpoint downloaded successfully"
                DETECTOR_CHECKPOINT="./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
            else
                echo "❌ Failed to download checkpoint"
                SKIP_EVALUATION_FINAL=true
            fi
        else
            echo "✅ Detection checkpoint found"
            DETECTOR_CHECKPOINT="./checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
        fi
    fi
fi

# Display evaluation settings
print_section "Evaluation Settings"

echo "GenEval Configuration:"
echo "  • Dataset: Official GenEval"
echo "  • Max prompts: ${MAX_PROMPTS:-all (~3000 prompts)}"
echo "  • Images per prompt: $GENERATION_NUMS"
echo "  • CFG scale: $CFG_SCALE"
echo "  • Random seed: $SEED"
echo ""

echo "Evaluation Thresholds:"
echo "  • Detection threshold: $THRESHOLD"
echo ""

# Default GenEval prompt file - you should download this
GENEVAL_PROMPT_FILE="./geneval_prompts.jsonl"

# Check if GenEval prompt file exists
if [ ! -f "$GENEVAL_PROMPT_FILE" ]; then
    echo "❌ ERROR: GenEval prompt file not found: $GENEVAL_PROMPT_FILE"
    echo ""
    echo "Please download the GenEval dataset prompts and save as:"
    echo "  $GENEVAL_PROMPT_FILE"
    echo ""
    echo "You can find the GenEval dataset at:"
    echo "  https://github.com/djghosh13/geneval"
    echo ""
    echo "For testing, you can use the example file:"
    echo "  cp geneval_prompts_example.jsonl geneval_prompts.jsonl"
    exit 1
fi

# Prepare evaluation arguments
GENEVAL_ARGS="--model_path \"$MODEL_PATH\" \
              --prompt_file \"$GENEVAL_PROMPT_FILE\" \
              --output_dir \"$OUTPUT_DIR\" \
              --device $DEVICE \
              --cfg_scale $CFG_SCALE \
              --generation_nums $GENERATION_NUMS \
              --conf_threshold $THRESHOLD \
              --exp_name \"$EXP_NAME\" \
              --report_to \"$REPORT_TO\" \
              --tracker_project_name \"$TRACKER_PROJECT\""

# Add detection checkpoint for evaluation
if [ "$SKIP_EVALUATION_FINAL" = false ] && [ "$GENERATE_ONLY" = false ]; then
    GENEVAL_ARGS="$GENEVAL_ARGS --detector_model_path \"$DETECTOR_CHECKPOINT\" \
                  --detector_config_path \"\""
elif [ "$GENERATE_ONLY" = true ]; then
    GENEVAL_ARGS="$GENEVAL_ARGS --generate_only"
fi

# Add max prompts limitation
if [ -n "$MAX_PROMPTS" ]; then
    echo "⚡ Using limited prompts for testing: $MAX_PROMPTS"
    # Note: You would need to modify the Python script to support max_prompts
else
    echo "⏳ Using full GenEval dataset"
fi

# Run GenEval evaluation
print_section "Running GenEval Evaluation"

if [ "$GENERATE_ONLY" = true ]; then
    echo "Generating images only..."
    echo "This will:"
    echo "  1. Load VILA-U model"
    echo "  2. Generate images for all GenEval prompts"
elif [ "$EVALUATE_ONLY" = false ]; then
    echo "Starting complete GenEval evaluation..."
    echo "This will:"
    echo "  1. Load VILA-U model"
    echo "  2. Generate images for all GenEval prompts"
    if [ "$SKIP_EVALUATION_FINAL" = false ]; then
        echo "  3. Evaluate compositional reasoning capabilities"
    fi
fi

echo ""

eval "python evaluate_geneval.py $GENEVAL_ARGS"

# Display results
print_section "GenEval Evaluation Results"

if [ "$GENERATE_ONLY" = true ]; then
    echo "✅ Image generation completed!"
    echo ""
    echo "Generated images saved to: $OUTPUT_DIR/generated_images/"
    echo ""
    echo "To run evaluation on generated images:"
    echo "  ./run_geneval_evaluation.sh --evaluate-only \"$OUTPUT_DIR/generated_images\""
    exit 0
fi

if [ "$SKIP_EVALUATION_FINAL" = true ]; then
    echo "⚠️  Evaluation step was skipped due to missing detection checkpoint"
    echo ""
    echo "To complete the evaluation:"
    echo "  1. Download the detection checkpoint manually or let the script download it"
    echo "  2. Run evaluation on generated images:"
    echo "     ./run_geneval_evaluation.sh --evaluate-only \"$OUTPUT_DIR/generated_images\""
    exit 0
fi

if [ -f "$OUTPUT_DIR/${EXP_NAME}_geneval_results.json" ]; then
    echo "🎯 GenEval Evaluation Completed Successfully!"
    echo ""
    python -c "
import json
data = json.load(open('$OUTPUT_DIR/${EXP_NAME}_geneval_results.json'))
print(f'🎯 Overall Accuracy: {data[\"overall_accuracy\"]:.3f}')
print()
print('📋 Task-specific Results:')
for key, value in data.items():
    if key.endswith('_accuracy') and key != 'overall_accuracy':
        task = key.replace('_accuracy', '').replace('_', ' ').title()
        print(f'  • {task:<15}: {value:.3f}')
print()
print('📁 Results Location:')
print(f'  • Generated Images: $OUTPUT_DIR/generated_images/')
print(f'  • Results: $OUTPUT_DIR/${EXP_NAME}_geneval_results.json')
"
    echo ""
    echo "✅ GenEval evaluation completed successfully!"
else
    echo "❌ GenEval evaluation failed - no results file found"
    echo "Check the error messages above for troubleshooting"
    exit 1
fi

# Performance guidance
echo ""
echo "📈 GenEval Score Interpretation:"
echo "  • > 0.8:  Excellent compositional understanding"
echo "  • 0.6-0.8: Good performance with room for improvement"
echo "  • 0.4-0.6: Moderate performance, significant issues"
echo "  • < 0.4:  Poor compositional understanding"
echo ""

# Task-specific guidance
echo "🧠 GenEval Task Breakdown:"
echo "  • Object Presence: Basic object recognition"
echo "  • Object Counting: Numerical understanding"
echo "  • Spatial Relations: Spatial relationship understanding"
echo ""

# Suggest next steps
echo "🔄 Next Steps:"
echo "  • Compare results with GenEval paper baselines"
echo "  • Examine failure cases in: $OUTPUT_DIR/${EXP_NAME}_geneval_results.json"
echo ""

echo "=== GenEval Evaluation Complete ==="