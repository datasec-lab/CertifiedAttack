import os
import numpy as np

# Define the directory to start the search
start_dir = '../experiments/attack/cifar100_post_RAND'
attack_list=[
    'untargeted/linf/score/{}/Bandit',
    'untargeted/linf/score/{}/NES',
    'untargeted/linf/score/{}/Parsimonious',
    'untargeted/linf/score/{}/Sign',
    'untargeted/linf/score/{}/Square',
    'untargeted/linf/score/{}/ZOSignSGD',

    'untargeted/linf/decision/{}/GeoDA',
    'untargeted/linf/decision/{}/HSJ',
    'untargeted/linf/decision/{}/Opt',
    'untargeted/linf/decision/{}/RayS',
    'untargeted/linf/decision/{}/SignFlip',
    'untargeted/linf/decision/{}/SignOPT',

    'untargeted/l2/score/{}/Bandit',
    'untargeted/l2/score/{}/NES',
    'untargeted/l2/score/{}/Simple',
    'untargeted/l2/score/{}/Square',
    'untargeted/l2/score/{}/ZOSignSGD',

    'untargeted/l2/decision/{}/Boundary',
    'untargeted/l2/decision/{}/GeoDA',
    'untargeted/l2/decision/{}/HSJ',
    'untargeted/l2/decision/{}/Opt',
    'untargeted/l2/decision/{}/SignOPT',

    'untargeted/unrestricted/decision/{}/PointWise',
    'untargeted/unrestricted/decision/{}/SparseEvo',
    'untargeted/unrestricted/decision/{}/CA_sssp',
    # 'untargeted/unrestricted/decision/{}/CA',


]

Models=['vgg','resnet','resnext','wrn']

# Function to find and print the contents of predictions.npz files
def find_and_print_npz_files(file_path):
    data = np.load(file_path, allow_pickle=True)
    # print(data)
    preds = data['preds']
    probs = data['probs']
    labels = data['labels']
    loss = data['loss']
    acc = data['acc']
    query = data['query']
    blacklight_detection_rate = data["bl_detect_rate"]
    blacklight_coverage = data["bl_coverage"]
    blacklight_query_to_detect = data["bl_q2detect"]
    # print(blacklight_detection_rate,blacklight_coverage,blacklight_query_to_detect)
    distance = data["distance"]
    splits=file_path.split('/')
    attack_name,model_name,attack_type,attack_norm,targeted,dataset=splits[-2],splits[-3],splits[-4],splits[-5],splits[-6],splits[-7]
    if "blacklight" in file_path:
        print(f"{dataset}|{targeted}|{attack_norm}|{attack_type}|{model_name}|{attack_name}: acc {acc*100:.1f} | query {query:.0f} | detection_rate {blacklight_detection_rate:.3f} | coverage {blacklight_coverage:.3f} | query to detect {blacklight_query_to_detect:.1f} | distance {distance:.2f}")
    else:
        print(f"{dataset}|{targeted}|{attack_norm}|{attack_type}|{model_name}|{attack_name}: acc {acc * 100:.1f} | query {query:.0f} | distance {distance:.2f}")

if __name__ == '__main__':

    for model in Models:
        for attack_path in attack_list:
            path=os.path.join(start_dir,attack_path.format(model),'predictions.npz')
            if os.path.exists(path):
                find_and_print_npz_files(path)
            else:
                splits = path.split('/')
                attack_name, model_name, attack_type, attack_norm, targeted, dataset = splits[-2], splits[-3], splits[-4], splits[-5], splits[-6], splits[-7]
                print(f"{dataset}|{targeted}|{attack_norm}|{attack_type}|{model_name}|{attack_name}: No Results")