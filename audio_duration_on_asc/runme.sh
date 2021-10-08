WORKSPACE= workspace_path
DATASET=data_set_path
PROFILE=train
BACKBONE=baseline # baseline or dresnet

# PRETRAIN=--pretrain
# MODEL_PATH='2021-09-28-11-46-03_BS_16_LR_0.0001_PF_t_BB_dres_AA_0.1_SL_5.0'
# PRETRAIN_MODEL_NAME='saved_model_312.pth'

BATCH_SIZE=16
LEARN_RATE=0.001
EPOCHS=800
ALPHA=0.1
SLICE=5.0


python3 dcase2020/main.py train --workspace=$WORKSPACE --dataset=$DATASET  --profile=$PROFILE --learning_rate=$LEARN_RATE --batch_size=$BATCH_SIZE --epochs=$EPOCHS --backbone=$BACKBONE $PRETRAIN --pretrain_path=$MODEL_PATH --pretrained_model_name=$PRETRAIN_MODEL_NAME --augmentation_alpha=$ALPHA --slice=$SLICE
