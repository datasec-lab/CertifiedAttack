################# imagenet_RAND VGG Untargeted ###############

nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/vgg_GeoDA.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_vgg_GeoDA.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/vgg_HSJ.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_vgg_HSJ.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/vgg_Opt.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_vgg_Opt.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/vgg_RayS.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_vgg_RayS.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/vgg_SignFlip.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_vgg_SignFlip.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/vgg_SignOPT.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_vgg_SignOPT.log &

nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/vgg_Bandit.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_score_vgg_Bandit.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/vgg_NES.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_linf_score_vgg_NES.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/vgg_Parsimonious.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_linf_score_vgg_Parsimonious.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/vgg_Sign.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_score_vgg_Sign.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/vgg_Square.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_linf_score_vgg_Square.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/vgg_ZOSignSGD.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_score_vgg_ZOSignSGD.log &

nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/decision/vgg_Boundary.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_l2_decision_vgg_Boundary.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/decision/vgg_GeoDA.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_l2_decision_vgg_GeoDA.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/decision/vgg_HSJ.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_l2_decision_vgg_HSJ.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/decision/vgg_Opt.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_l2_decision_vgg_Opt.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/decision/vgg_SignOPT.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_l2_decision_vgg_SignOPT.log &

nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/score/vgg_Bandit.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_l2_score_vgg_Bandit.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/score/vgg_NES.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_l2_score_vgg_NES.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/score/vgg_Simple.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_l2_score_vgg_Simple.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/score/vgg_Square.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_l2_score_vgg_Square.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/score/vgg_ZOSignSGD.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_l2_score_vgg_ZOSignSGD.log &

nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/unrestricted/vgg_SparseEvo.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_lp_unrestricted_vgg_SparseEvo.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/unrestricted/vgg_PointWise.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_lp_unrestricted_vgg_PointWise.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/unrestricted/vgg_CertifiedAttack.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_lp_unrestricted_vgg_CertifiedAttack.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/unrestricted/vgg_CertifiedAttack_sssp.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_lp_unrestricted_vgg_CertifiedAttack_sssp.log &


################# imagenet_RAND ResNet Untargeted ###############

nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/resnet_GeoDA.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_resnet_GeoDA.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/resnet_HSJ.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_resnet_HSJ.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/resnet_Opt.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_resnet_Opt.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/resnet_RayS.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_resnet_RayS.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/resnet_SignFlip.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_resnet_SignFlip.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/decision/resnet_SignOPT.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_linf_decision_resnet_SignOPT.log &

nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/resnet_Bandit.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_score_resnet_Bandit.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/resnet_NES.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_score_resnet_NES.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/resnet_Parsimonious.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_score_resnet_Parsimonious.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/resnet_Sign.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_score_resnet_Sign.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/resnet_Square.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_score_resnet_Square.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/linf/score/resnet_ZOSignSGD.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_linf_score_resnet_ZOSignSGD.log &

nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/decision/resnet_Boundary.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_l2_decision_resnet_Boundary.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/decision/resnet_GeoDA.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_l2_decision_resnet_GeoDA.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/decision/resnet_HSJ.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_l2_decision_resnet_HSJ.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/decision/resnet_Opt.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_l2_decision_resnet_Opt.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/decision/resnet_SignOPT.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_l2_decision_resnet_SignOPT.log &

nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/score/resnet_Bandit.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_l2_score_resnet_Bandit.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/score/resnet_NES.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_l2_score_resnet_NES.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/score/resnet_Simple.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_l2_score_resnet_Simple.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/score/resnet_Square.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_l2_score_resnet_Square.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/l2/score/resnet_ZOSignSGD.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_l2_score_resnet_ZOSignSGD.log &

nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/unrestricted/resnet_SparseEvo.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_lp_unrestricted_resnet_SparseEvo.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/unrestricted/resnet_PointWise.yaml" device cuda:0 >./nohup_logs/imagenet_RAND_untargeted_lp_unrestricted_resnet_PointWise.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/unrestricted/resnet_CertifiedAttack.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_lp_unrestricted_resnet_CertifiedAttack.log &
nohup python attack.py --config "./configs/attack/imagenet_RAND/untargeted/unrestricted/resnet_CertifiedAttack_sssp.yaml" device cuda:1 >./nohup_logs/imagenet_RAND_untargeted_lp_unrestricted_resnet_CertifiedAttack_sssp.log &
