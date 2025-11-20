#!/bin/bash
# Demo script for kymo-tracker: modular stages for training and inference

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Default values
STAGES="all"
DATA_DIR="demo/data"
MODEL_PATH="artifacts/demo_model.pth"
CLASSICAL_DIR="demo/results/classical"
DEEPLEARNING_DIR="demo/results/deeplearning"
OUTPUT_DIR="demo/results"

# Parse command line arguments
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [STAGES...]

Run kymo-tracker demo stages. If no stages are specified, runs all stages.

Stages:
  1  generate_data    - Generate synthetic test cases
  2  train_model       - Train the deep learning model
  3  run_classical     - Run classical inference pipeline
  4  run_deeplearning   - Run deep learning inference pipeline
  5  visualize         - Create comparison plots
  all                 - Run all stages (default)

Options:
  -h, --help           Show this help message
  --data-dir DIR       Directory for test cases (default: demo/data)
  --model-path PATH    Path to trained model (default: artifacts/demo_model.pth)
  --classical-dir DIR  Directory for classical results (default: demo/results/classical)
  --dl-dir DIR         Directory for deep learning results (default: demo/results/deeplearning)
  --output-dir DIR     Directory for output plots (default: demo/results)

Examples:
  $0                                    # Run all stages
  $0 1 2                                # Only generate data and train model
  $0 3 4 5                              # Only run inference and visualization
  $0 2 --model-path my_model.pth        # Train with custom model path
EOF
}

# Parse arguments
STAGE_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --classical-dir)
            CLASSICAL_DIR="$2"
            shift 2
            ;;
        --dl-dir)
            DEEPLEARNING_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            STAGE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Determine which stages to run
if [[ ${#STAGE_ARGS[@]} -eq 0 ]]; then
    STAGES="all"
else
    STAGES="${STAGE_ARGS[*]}"
fi

echo "=========================================="
echo "Kymo-Tracker Demo Script"
echo "=========================================="
echo ""
echo "Stages to run: $STAGES"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    echo "Visit: https://github.com/astral-sh/uv"
    exit 1
fi

# Ensure dependencies are installed
echo "Ensuring dependencies are installed..."
uv sync
echo ""

# Function to run a stage
run_stage() {
    local stage_num=$1
    local stage_name=$2
    shift 2
    local extra_args=("$@")
    
    echo "=========================================="
    echo "STAGE $stage_num: $stage_name"
    echo "=========================================="
    echo ""
    
    case $stage_num in
        1)
            uv run python demo/generate_data.py --output-dir "$DATA_DIR" "${extra_args[@]}"
            ;;
        2)
            uv run python demo/train_model.py --model-path "$MODEL_PATH" "${extra_args[@]}"
            ;;
        3)
            uv run python demo/run_classical.py --data-dir "$DATA_DIR" --output-dir "$CLASSICAL_DIR" "${extra_args[@]}"
            ;;
        4)
            uv run python demo/run_deeplearning.py --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --output-dir "$DEEPLEARNING_DIR" "${extra_args[@]}"
            ;;
        5)
            uv run python demo/visualize.py --data-dir "$DATA_DIR" --classical-dir "$CLASSICAL_DIR" --deeplearning-dir "$DEEPLEARNING_DIR" --output-dir "$OUTPUT_DIR" "${extra_args[@]}"
            ;;
        *)
            echo "Unknown stage: $stage_num"
            exit 1
            ;;
    esac
    
    echo ""
}

# Run stages
if [[ "$STAGES" == "all" ]]; then
    run_stage 1 "Generate Data"
    run_stage 2 "Train Model" --skip-if-exists
    run_stage 3 "Run Classical Pipeline"
    run_stage 4 "Run Deep Learning Pipeline"
    run_stage 5 "Visualize Comparison"
else
    for stage in $STAGES; do
        case $stage in
            1|generate_data)
                run_stage 1 "Generate Data"
                ;;
            2|train_model)
                run_stage 2 "Train Model"
                ;;
            3|run_classical)
                run_stage 3 "Run Classical Pipeline"
                ;;
            4|run_deeplearning)
                run_stage 4 "Run Deep Learning Pipeline"
                ;;
            5|visualize)
                run_stage 5 "Visualize Comparison"
                ;;
            *)
                echo "Unknown stage: $stage"
                echo "Valid stages: 1, 2, 3, 4, 5, or 'all'"
                exit 1
                ;;
        esac
    done
fi

echo "=========================================="
echo "Demo completed successfully!"
echo "=========================================="
echo ""
echo "Check the following directories for results:"
echo "  - $DATA_DIR          : Test case data"
echo "  - $CLASSICAL_DIR     : Classical results"
echo "  - $DEEPLEARNING_DIR  : Deep learning results"
echo "  - $OUTPUT_DIR        : Comparison plots"
echo "  - $(dirname "$MODEL_PATH")        : Trained model"
echo ""
