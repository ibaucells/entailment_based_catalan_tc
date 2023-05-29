BASE_DIR='.'
MODEL_NAME_OR_PATH='symanto/xlm-roberta-base-snli-mnli-anli-xnli'

DATASET_LOADER_PATH=$BASE_DIR'/data/dataloader_nli.py'
LR=0.00005
SEED=26

#MODEL ARGUMENTS

MODEL=$MODEL_NAME_OR_PATH
MODEL_NAME=$( basename $MODEL )
NUM_EPOCHS=1
BATCH_SIZE=8
MAX_LENGTH=512
GRADIENT_ACC_STEPS=1
BATCH_SIZE_PER_GPU=$(( $BATCH_SIZE*$GRADIENT_ACC_STEPS ))
LEARN_RATE=$LR
DATASET_PATH=$DATASET_LOADER_PATH
DATASET_NAME=$( basename $DATASET_PATH )


TASK="nli"
METRIC="f1_weighted_cls"
SCRIPT=$BASE_DIR"/run_tc.py"


MODEL_ARGS=""
if test ! -z $TASK; then
    MODEL_ARGS="--task_name $TASK "
fi
MODEL_ARGS+=" \
 --model_name_or_path $MODEL \
 --dataset_name $DATASET_PATH \
 --do_train \
 --do_eval \
 --do_predict \
 --evaluation_strategy epoch \
 --num_train_epochs $NUM_EPOCHS \
 --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
 --per_device_train_batch_size $BATCH_SIZE \
 --learning_rate $LEARN_RATE \
 --max_seq_length $MAX_LENGTH \
 --metric_for_best_model $METRIC \
 --save_strategy epoch \
 "

#OUTPUT ARGUMENTS

OUTPUT_DIR=$BASE_DIR
LOGGING_DIR=$OUTPUT_DIR/'logs'
DIR_NAME=$ft_${BATCH_SIZE_PER_GPU}_${LEARN_RATE}_$DATETIME
CACHE_DIR=$OUTPUT_DIR/$DIR_NAME'/cache/'

OUTPUT_ARGS=" \
 --output_dir $OUTPUT_DIR/$DIR_NAME \
 --overwrite_output_dir \
 --logging_dir $LOGGING_DIR/$DIR_NAME \
 --logging_strategy epoch \
 --cache_dir $CACHE_DIR \
 --overwrite_cache \
 --load_best_model_at_end \
 "

python $SCRIPT --seed $SEED $MODEL_ARGS $OUTPUT_ARGS
