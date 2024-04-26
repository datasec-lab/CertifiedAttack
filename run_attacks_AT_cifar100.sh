################# cifar100_AT ResNet Untargeted ###############

nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/decision/resnet_GeoDA.yaml" device cuda:0 >./nohup_logs/cifar100_AT_untargeted_linf_decision_resnet_GeoDA.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/decision/resnet_HSJ.yaml" device cuda:1 >./nohup_logs/cifar100_AT_untargeted_linf_decision_resnet_HSJ.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/decision/resnet_Opt.yaml" device cuda:2 >./nohup_logs/cifar100_AT_untargeted_linf_decision_resnet_Opt.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/decision/resnet_RayS.yaml" device cuda:3 >./nohup_logs/cifar100_AT_untargeted_linf_decision_resnet_RayS.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/decision/resnet_SignFlip.yaml" device cuda:0 >./nohup_logs/cifar100_AT_untargeted_linf_decision_resnet_SignFlip.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/decision/resnet_SignOPT.yaml" device cuda:1 >./nohup_logs/cifar100_AT_untargeted_linf_decision_resnet_SignOPT.log &

nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/score/resnet_Bandit.yaml" device cuda:2 >./nohup_logs/cifar100_AT_untargeted_linf_score_resnet_Bandit.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/score/resnet_NES.yaml" device cuda:3 >./nohup_logs/cifar100_AT_untargeted_linf_score_resnet_NES.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/score/resnet_Parsimonious.yaml" device cuda:0 >./nohup_logs/cifar100_AT_untargeted_linf_score_resnet_Parsimonious.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/score/resnet_Sign.yaml" device cuda:1 >./nohup_logs/cifar100_AT_untargeted_linf_score_resnet_Sign.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/score/resnet_Square.yaml" device cuda:2 >./nohup_logs/cifar100_AT_untargeted_linf_score_resnet_Square.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/linf/score/resnet_ZOSignSGD.yaml" device cuda:3 >./nohup_logs/cifar100_AT_untargeted_linf_score_resnet_ZOSignSGD.log &

nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/l2/decision/resnet_Boundary.yaml" device cuda:0 >./nohup_logs/cifar100_AT_untargeted_l2_decision_resnet_Boundary.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/l2/decision/resnet_GeoDA.yaml" device cuda:1 >./nohup_logs/cifar100_AT_untargeted_l2_decision_resnet_GeoDA.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/l2/decision/resnet_HSJ.yaml" device cuda:2 >./nohup_logs/cifar100_AT_untargeted_l2_decision_resnet_HSJ.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/l2/decision/resnet_Opt.yaml" device cuda:3 >./nohup_logs/cifar100_AT_untargeted_l2_decision_resnet_Opt.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/l2/decision/resnet_SignOPT.yaml" device cuda:0 >./nohup_logs/cifar100_AT_untargeted_l2_decision_resnet_SignOPT.log &

nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/l2/score/resnet_Bandit.yaml" device cuda:1 >./nohup_logs/cifar100_AT_untargeted_l2_score_resnet_Bandit.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/l2/score/resnet_NES.yaml" device cuda:2 >./nohup_logs/cifar100_AT_untargeted_l2_score_resnet_NES.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/l2/score/resnet_Simple.yaml" device cuda:3 >./nohup_logs/cifar100_AT_untargeted_l2_score_resnet_Simple.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/l2/score/resnet_Square.yaml" device cuda:0 >./nohup_logs/cifar100_AT_untargeted_l2_score_resnet_Square.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/l2/score/resnet_ZOSignSGD.yaml" device cuda:1 >./nohup_logs/cifar100_AT_untargeted_l2_score_resnet_ZOSignSGD.log &

nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/unrestricted/resnet_SparseEvo.yaml" device cuda:0 >./nohup_logs/cifar100_AT_untargeted_lp_unrestricted_resnet_SparseEvo.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/unrestricted/resnet_PointWise.yaml" device cuda:0 >./nohup_logs/cifar100_AT_untargeted_lp_unrestricted_resnet_PointWise.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/unrestricted/resnet_CertifiedAttack.yaml" device cuda:1 >./nohup_logs/cifar100_AT_untargeted_lp_unrestricted_resnet_CertifiedAttack.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/unrestricted/resnet_CertifiedAttack_sssp.yaml" device cuda:2 >./nohup_logs/cifar100_AT_untargeted_lp_unrestricted_resnet_CertifiedAttack_sssp.log &

nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/unrestricted/resnet_SparseEvo_l2.yaml" device cuda:0 >./nohup_logs/cifar100_AT_untargeted_lp_unrestricted_resnet_SparseEvo_l2.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/unrestricted/resnet_PointWise_l2.yaml" device cuda:0 >./nohup_logs/cifar100_AT_untargeted_lp_unrestricted_resnet_PointWise_l2.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/unrestricted/resnet_CertifiedAttack_l2.yaml" device cuda:3 >./nohup_logs/cifar100_AT_untargeted_lp_unrestricted_resnet_CertifiedAttack_l2.log &
nohup python attack.py --config "./configs/attack/cifar100_AT/untargeted/unrestricted/resnet_CertifiedAttack_sssp_l2.yaml" device cuda:3 >./nohup_logs/cifar100_AT_untargeted_lp_unrestricted_resnet_CertifiedAttack_sssp_l2.log &
