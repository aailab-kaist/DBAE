DATASET_NAME=$1
PRED=$2
STO=$3
CKPT=$4

source ./args.sh $DATASET_NAME $PRED $STO

FREQ_SAVE_ITER=10000
NGPU=4
ATTN_TYPE="non_flash"

CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n $NGPU --mca btl vader,self python3 dbae_train.py --exp=$EXP \
 --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
  --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
   --image_size $IMG_SIZE --lr 0.0001 --num_channels $NUM_CH --num_head_channels 64 --latent_dim=$LATENT --sto=$STO --end=$END --end_data=$ENDDATA \
    --num_res_blocks $NUM_RES_BLOCKS --resblock_updown True --microbatch 32 ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
     --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
    --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 --weight_schedule bridge_karras \
     ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
      --data_dir=$DATA_DIR --dataset=$DATASET ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
      --num_workers=8  --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
      --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER --debug=False \
      ${CKPT:+ --resume_checkpoint="${CKPT}"}