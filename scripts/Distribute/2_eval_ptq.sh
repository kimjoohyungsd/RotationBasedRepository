# 1. 인자 확인 (인자가 2개가 아니면 사용법을 출력하고 종료)
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <CUDA_VISIBLE_DEVICES> <MODEL_PATH>"
    echo "Example: $0 0,1,2,3 meta-llama/Llama-2-7b-hf"
    exit 1
fi

cleanup() {
    echo ""
    echo "!!! Keyboard Interrupt detected. Terminating python processes... !!!"
    # 현재 스크립트가 실행한 자식 프로세스 그룹을 종료
    # 'jobs -p'로 현재 실행 중인 자식 PID를 받아와서 죽입니다.
    pkill -P $$ 
    exit 1
}

trap cleanup SIGINT
# 2. 인자 할당
# $1은 첫 번째 인자 (GPU 번호), $2는 두 번째 인자 (모델 경로)
CUDA_DEVICES=$1
MODEL_PATH=$2
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="/home/jhkcool97/RotationBasedRepository/outputs/${MODEL_NAME}"
<<<<<<< HEAD

W_BIT=4  # Weight bit-width
A_BIT=4  # Activation bit-width
=======
W_BIT=4  # Weight bit-width
A_BIT=8  # Activation bit-width
>>>>>>> eeb12376a9388aa382db01a1623266ad6c51621d
# 디렉토리 생성 확인
mkdir -p "$OUTPUT_DIR"

# -------------------------------------------------------------------------
# 실험 1: 모든 Option 실행
echo "Starting Experiment 1: R4 Option on GPUs $CUDA_DEVICE.S"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python ptq.py \
    --input_model "$MODEL_PATH" \
    --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
    --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    --rotate --distribute --online_r2 --wikitext2 --w_rtn --w_clip \
<<<<<<< HEAD
    > "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_All_Rotations.txt" 2>&1
=======
    > "${OUTPUT_DIR}/log__W${W_BIT}A${A_BIT}_ALL_ROTATIONS.txt" 2>&1
>>>>>>> eeb12376a9388aa382db01a1623266ad6c51621d

echo "Experiment 1 finished. Starting Experiment 2: No R4 Option"

# 실험 2: No R4 Option 실행
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python ptq.py \
    --input_model "$MODEL_PATH" \
    --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
<<<<<<< HEAD
    --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    --rotate --distribute --online_r2 --deactivate_r4 --wikitext2 --w_rtn --w_clip \
    > "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_No_R4.txt" 2>&1
=======
    --w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 \
    --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    --rotate --distribute --online_r2 --deactivate_r4 --wikitext2 --w_rtn --w_clip \
    > "${OUTPUT_DIR}/log__W${W_BIT}A${A_BIT}_No_R4.txt" 2>&1
>>>>>>> eeb12376a9388aa382db01a1623266ad6c51621d

# 실험 3: No R2 Option 실행
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python ptq.py \
    --input_model "$MODEL_PATH" \
    --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
<<<<<<< HEAD
    --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    --rotate --distribute --deactivate_r2 --wikitext2 --w_rtn --w_clip \
    > "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_No_R2.txt" 2>&1
=======
    --w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 \
    --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    --rotate --distribute --deactivate_r2 --wikitext2 --w_rtn --w_clip \
    > "${OUTPUT_DIR}/log__W${W_BIT}A${A_BIT}_No_R2.txt" 2>&1
>>>>>>> eeb12376a9388aa382db01a1623266ad6c51621d

# 실험 4: No R1 Option 실행
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python ptq.py \
    --input_model "$MODEL_PATH" \
    --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
    --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    --rotate --distribute --deactivate_r1 --online_r2 --wikitext2 --w_rtn --w_clip \
<<<<<<< HEAD
    > "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_No_R1.txt" 2>&1
=======
    > "${OUTPUT_DIR}/log__W${W_BIT}A${A_BIT}_No_R1.txt" 2>&1
>>>>>>> eeb12376a9388aa382db01a1623266ad6c51621d
