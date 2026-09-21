REPO="/home/jhkcool97/RotationBasedRepository"
cd "$REPO" || exit 1

cleanup() {
    echo ""
    echo "!!! Keyboard Interrupt detected. Terminating python processes... !!!"
    pkill -P $$
    exit 1
}
trap cleanup SIGINT

# "<HF model id>|<CUDA_VISIBLE_DEVICES>"
JOBS=(
    "meta-llama/Llama-2-7b-hf|0,1"
    "meta-llama/Llama-2-13b-hf|2,3,4"
    "meta-llama/Llama-3.1-8B|5,6"
)