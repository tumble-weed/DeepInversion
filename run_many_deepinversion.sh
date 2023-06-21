PYTHON="python"
# PYTHON="python -m ipdb -c c"

# visible but sharp
# pr_inversion/best_images/output_00030_gpu_0.png
# USE_PR=1 CUDA_VISIBLE_DEVICES=1 python -m ipdb -c c imagenet_inversion.py --bs=84 --do_flip --exp_name="pr_inversion" --r_feature=0.01 --arch_name="mobilenet_v2" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25

# invisible
# pr_inversion_pr_1e-3_bn_0/best_images/output_00030_gpu_0.png
# USE_PR=1 PR_LOSS=1e-3 CUDA_VISIBLE_DEVICES=1 $PYTHON imagenet_inversion.py --bs=84 --do_flip --exp_name="pr_inversion_pr_1e-3_bn_0" --r_feature=0.0 --arch_name="mobilenet_v2" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25

# visible but sharp
# pr_inversion_pr_1e0_bn_1e-3/best_images/output_00030_gpu_0.png
# USE_PR=1 CUDA_VISIBLE_DEVICES=1 $PYTHON imagenet_inversion.py --bs=84 --do_flip --exp_name="pr_inversion_pr_1e0_bn_1e-3" --r_feature=0.001 --arch_name="mobilenet_v2" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25

# visible but sharp
# pr_inversion_pr_1e-1_bn_1e-2/best_images/output_00030_gpu_0.png
# USE_PR=1  PR_LOSS=1e-1 CUDA_VISIBLE_DEVICES=1 $PYTHON imagenet_inversion.py --bs=84 --do_flip --exp_name="pr_inversion_pr_1e-1_bn_1e-2" --r_feature=0.01 --arch_name="mobilenet_v2" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25


# not disappeared, less sharp but still
# pr_inversion_pr_1e-1_bn_1e-2_lr1_22e-4/best_images/output_00030_gpu_0.png
# LR1:sh LR2:vertsparams 
# default: LR1=22e-3, LR2=8e-4
# USE_PR=1  PR_LOSS=1e-1 LR1=22e-4  CUDA_VISIBLE_DEVICES=1 $PYTHON imagenet_inversion.py --bs=84 --do_flip --exp_name="pr_inversion_pr_1e-1_bn_1e-2_lr1_22e-4" --r_feature=0.01 --arch_name="mobilenet_v2" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25


# incomplete
# pr_inversion_pr_1e-1_bn_1e-2_lr1_22e-5/best_images/output_00030_gpu_0.png
# LR1:sh LR2:vertsparams 
# default: LR1=22e-3, LR2=8e-4
# USE_PR=1  PR_LOSS=1e-1 LR1=22e-5  CUDA_VISIBLE_DEVICES=1 $PYTHON imagenet_inversion.py --bs=84 --do_flip --exp_name="pr_inversion_pr_1e-1_bn_1e-2_lr1_22e-5" --r_feature=0.01 --arch_name="mobilenet_v2" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25


# 
# pr_inversion_pr_1e-1_bn_1e-2_lr1_22e-5/best_images/output_00030_gpu_0.png
# LR1:sh LR2:vertsparams 
# default: LR1=22e-3, LR2=8e-4
# ENABLE_PR=0 USE_PR=1  PR_LOSS=1e-1 LR1=22e-4  CUDA_VISIBLE_DEVICES=1 $PYTHON imagenet_inversion.py --bs=84 --do_flip --exp_name="pr_inversion_pr_1e-1_bn_1e-2_lr1_22e-4_disable_pr" --r_feature=0.01 --arch_name="mobilenet_v2" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25



# incomplete pr
# pr_inversion_pr_1e-1_bn_1e-2_lr1_22e-5/best_images/output_00030_gpu_0.png
# LR1:sh LR2:vertsparams 
# default: LR1=22e-3, LR2=8e-4
# ENABLE_PR=0 USE_PR=1  PR_LOSS=1e-1 LR1=22e-4  CUDA_VISIBLE_DEVICES=1 $PYTHON imagenet_inversion.py --bs=84 --do_flip --exp_name="pr_inversion_rn50_pr_1e-1_bn_1e-2_lr1_22e-4_disable_pr" --r_feature=0.01 --arch_name="resnet50" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25



# sharp pr
# pr_inversion_pr_1e-1_bn_1e-2_lr1_22e-5/best_images/output_00030_gpu_0.png
# LR1:sh LR2:vertsparams 
# default: LR1=22e-3, LR2=8e-4
# LR1="22e-3"
# USE_PR=1  PR_LOSS=1e-1 LR1=$LR1  CUDA_VISIBLE_DEVICES=1 $PYTHON imagenet_inversion.py --bs=84 --do_flip --exp_name="pr_inversion_rn50_pr_1e-1_bn_1e-2_lr1_$LR1" --r_feature=0.01 --arch_name="resnet50" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25



# how do i find:
# camera distance from the pose matrix
# camera focal lengths from the pose matrix

# ==========================================================================
# generic caller
# ==========================================================================

# # setting 1
# USE_PR="1"
# PR_LOSS="1e-1"
# LR1="22e-4"
# bs=84
# model="resnet50"
# bn="1e-2"
#-----------------------------------------------------------
# # No PR loss, only bn and smoothness
# # totally died out ()
# USE_PR="1"
# PR_LOSS="0"
# LR1="22e-4"
# bs=84
# model="resnet50"
# bn="1e-2"
# # #-----------------------------------------------------------
# # High PR loss, moderate LR
# # totally died out ( with random pose)
# USE_PR="1"
# PR_LOSS="1e0"
# LR1="22e-4"
# bs=84
# model="resnet50"
# bn="1e-2"
# # #-----------------------------------------------------------
# # High PR loss, moderate LR (incomplete)
# USE_PR="1"
# PR_LOSS="1e0"
# LR1="22e-4"
# bs=84
# model="resnet50"
# bn="1e-2"
# # #-----------------------------------------------------------
# # High PR loss, high LR (really noisy)
# USE_PR="1"
# PR_LOSS="1e0"
# LR1="22e-3"
# bs=84
# model="resnet50"
# bn="1e-2"
# # #-----------------------------------------------------------
# random pose High PR loss, high LR (?)
# USE_PR="1"
# PR_LOSS="1e0"
# LR1="22e-3"
# bs=84
# model="resnet50"
# bn="1e-2"
# export TARGET_POSE2="1"
# #-----------------------------------------------------------
# USE_PR="1"
# PR_LOSS="1e-1"
# LR1="22e-4"
# bs=84
# model="resnet50"
# bn="1e-2"
# #-----------------------------------------------------------
# USE_FIXED_DIR="1"
# USE_PR="1"
# PR_LOSS="1e0"
# #22e-3
# LR1="22e-4"
# #8e-4
# LR2="8e-5"
# bs=84
# model="resnet50"
# bn="1e-2"
# export TARGET_POSE2="1"


# # if model is resnet50, make the modelstub rn50
# if [ "$model" = "resnet50" ]; then
#    modelstub="rn50"
# else
#    modelstub=$model
# fi
# # modelstub
# USE_FIXED_DIR=$USE_FIXED_DIR ONLYBASE=1 USE_PR=$USE_PR  PR_LOSS=$PR_LOSS LR1=$LR1  CUDA_VISIBLE_DEVICES=1 $PYTHON imagenet_inversion.py --bs=$bs --do_flip --exp_name="pr_inversion_${modelstub}_pr_${PR_LOSS}_bn_${bn}_lr1_${LR1}_lr2_${LR2}" --r_feature=$bn --arch_name="$model" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25
# #-----------------------------------------------------------
PR_CLASS="25"
USE_PR="1"
PR_LOSS="1e0"
#22e-3
LR1="22e-3"
#8e-4
LR2="8e-4"
bs=84
model="resnet50"
bn="1e-2"
export TARGET_POSE2="1"




# if model is resnet50, make the modelstub rn50
if [ "$model" = "resnet50" ]; then
  modelstub="rn50"
else
  modelstub=$model
fi
# modelstub
PR_CLASS=$PR_CLASS ONLYBASE=1 USE_PR=$USE_PR  PR_LOSS=$PR_LOSS LR1=$LR1  CUDA_VISIBLE_DEVICES=1 $PYTHON imagenet_inversion.py --bs=$bs --do_flip --exp_name="pr_inversion_${modelstub}_pr_${PR_LOSS}_bn_${bn}_lr1_${LR1}_lr2_${LR2}_class${PR_CLASS}" --r_feature=$bn --arch_name="$model" --verifier --adi_scale=0.0 --setting_id=2 --lr 0.25