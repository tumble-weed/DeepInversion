PYTHON="python"
# (
# #TV_L2=0.0001;
# TV_L2=0.0001;
# #BN=0.01;
# BN=0.1;
# python facenet_inversion.py --bs=84 --do_flip --exp_name="facenet_inversion_tv${TV_L2}_bn${BN}" --r_feature=$BN --arch_name="facenet" --verifier --adi_scale=0.0 --setting_id=1 --lr 0.25 --tv_l2 $TV_L2
# )

#==============================================================================
USE_PR="1"
PR_LOSS="1e0"
#22e-3
LR1="22e-3"
#8e-4
LR2="8e-4"
bs=84


#0.01
bn="1e-1"
USE_FIXED_DIR=1



USE_FIXED_DIR=$USE_FIXED_DIR ONLYBASE=1 USE_PR=$USE_PR  PR_LOSS=$PR_LOSS LR1=$LR1  CUDA_VISIBLE_DEVICES=1 $PYTHON facenet_inversion.py --bs=$bs --do_flip --exp_name="facenet_pr_inversion_pr_${PR_LOSS}_bn_${bn}_lr1_${LR1}_lr2_${LR2}" --r_feature=$bn --verifier --adi_scale=0.0 --setting_id=2 --lr 0.25
#=====================================================================================
# USE_PR="1"
# PR_LOSS="1e0"
# #22e-3
# LR1="22e-3"
# #8e-4
# LR2="8e-4"
# bs=84


# #0.01
# bn="1e-1"

# AZIM_MAG=5
# ELEV_MAG=0
# DIST_MAG=1.2


# AZIM_MAG=$AZIM_MAG ELEV_MAG=$ELEV_MAG DIST_MAG=$DIST_MAG ONLYBASE=1 USE_PR=$USE_PR  PR_LOSS=$PR_LOSS LR1=$LR1  CUDA_VISIBLE_DEVICES=1 $PYTHON facenet_inversion.py --bs=$bs --do_flip --exp_name="facenet_pr_inversion_pr_${PR_LOSS}_bn_${bn}_lr1_${LR1}_lr2_${LR2}_AZIM${AZIM_MAG}_ELEV${ELEV_MAG}_DIST${DIST_MAG}" --r_feature=$bn --verifier --adi_scale=0.0 --setting_id=2 --lr 0.25