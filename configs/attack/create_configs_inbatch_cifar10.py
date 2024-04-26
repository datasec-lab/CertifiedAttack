import os
import yaml
def create_post_RAND_configs_from_vgg(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') and 'vgg' in file:
                original_file_path = os.path.join(subdir, file)
                # Generate a new file path for ResNet configuration
                new_file_path = original_file_path.replace('cifar10', 'cifar10_post_RAND')
                post_RAND_modify_and_save_new_config_vgg(original_file_path, new_file_path)
                print(f'Original: {original_file_path} -> New: {new_file_path}')

def post_RAND_modify_and_save_new_config_vgg(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config['attack']['p']=='linf':
        config['attack']['epsilon']=25.5
    if config['attack']['p']=='l2':
        config['attack']['epsilon'] = 1275.0

    config['defense']['post_sigma']=0.2

    config['test']['output_dir'] = config['test']['output_dir'].replace('cifar10', 'cifar10_post_RAND')
    config['test']['batch_size'] = 1
    # Ensure the directory for the new file path exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Save the modified config to a new file
    with open(new_file_path, 'w') as file:
        yaml.dump(config, file)

def create_RAND_configs_from_vgg(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') and 'vgg' in file:
                original_file_path = os.path.join(subdir, file)
                # Generate a new file path for ResNet configuration
                new_file_path = original_file_path.replace('cifar10', 'cifar10_RAND')
                RAND_modify_and_save_new_config_vgg(original_file_path, new_file_path)
                print(f'Original: {original_file_path} -> New: {new_file_path}')

def RAND_modify_and_save_new_config_vgg(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config['attack']['p']=='linf':
        config['attack']['epsilon']=25.5
    if config['attack']['p']=='l2':
        config['attack']['epsilon'] = 1275.0

    config['defense']['sigma']=0.02

    config['test']['output_dir'] = config['test']['output_dir'].replace('cifar10', 'cifar10_RAND')
    config['test']['batch_size'] = 1
    # Ensure the directory for the new file path exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Save the modified config to a new file
    with open(new_file_path, 'w') as file:
        yaml.dump(config, file)
def create_blacklight_configs_from_vgg(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') and 'vgg' in file:
                original_file_path = os.path.join(subdir, file)
                # Generate a new file path for ResNet configuration
                new_file_path = original_file_path.replace('cifar10', 'cifar10_blacklight')
                blacklight_modify_and_save_new_config_vgg(original_file_path, new_file_path)
                print(f'Original: {original_file_path} -> New: {new_file_path}')

def blacklight_modify_and_save_new_config_vgg(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config['attack']['p']=='linf':
        config['attack']['epsilon']=25.5
    if config['attack']['p']=='l2':
        config['attack']['epsilon'] = 1275.0

    config['defense']['blacklight']=True

    config['test']['output_dir'] = config['test']['output_dir'].replace('cifar10', 'cifar10_blacklight')
    config['test']['batch_size'] = 1
    # Ensure the directory for the new file path exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Save the modified config to a new file
    with open(new_file_path, 'w') as file:
        yaml.dump(config, file)
def modify_and_save_new_config_resnet(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Modify the model section for ResNet
    config['model'] = {
        'type': 'cifar',
        'name': 'resnet',
        'init_mode': 'kaiming_fan_out',
        'resnet': {
            'depth': 110,
            'initial_channels': 16,
            'block_type': 'basic'
        },
        'normalize_layer': True
    }

    # Modify the test section for ResNet
    config['test']['checkpoint'] = config['test']['checkpoint'].replace('vgg', 'resnet')
    config['test']['output_dir'] = config['test']['output_dir'].replace('vgg', 'resnet')

    # Ensure the directory for the new file path exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Save the modified config to a new file
    with open(new_file_path, 'w') as file:
        yaml.dump(config, file)


def create_resnet_configs_from_vgg(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') and 'vgg' in file:
                original_file_path = os.path.join(subdir, file)
                # Generate a new file path for ResNet configuration
                new_file_path = original_file_path.replace('vgg', 'resnet')
                modify_and_save_new_config_resnet(original_file_path, new_file_path)
                print(f'Original: {original_file_path} -> New: {new_file_path}')

def modify_and_save_new_config_resnext(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Modify the model section for ResNet
    config['model'] = {
        'type': 'cifar',
        'name': 'resnext',
        'init_mode': 'kaiming_fan_out',
        'resnext': {
            'depth': 29,
            'initial_channels': 64,
            'cardinality': 8,
            'base_channels': 64
        },
        'normalize_layer': True
    }

    # Modify the test section for ResNet
    config['test']['checkpoint'] = "experiments/cifar10/resnext/exp00/checkpoint_00300.pth"
    config['test']['output_dir'] = config['test']['output_dir'].replace('vgg', 'resnext')

    # Ensure the directory for the new file path exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Save the modified config to a new file
    with open(new_file_path, 'w') as file:
        yaml.dump(config, file)

def create_resnext_configs_from_vgg(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') and 'vgg' in file:
                original_file_path = os.path.join(subdir, file)
                # Generate a new file path for ResNet configuration
                new_file_path = original_file_path.replace('vgg', 'resnext')
                modify_and_save_new_config_resnext(original_file_path, new_file_path)
                print(f'Original: {original_file_path} -> New: {new_file_path}')

def modify_and_save_new_config_wrn(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Modify the model section for ResNet
    config['model'] = {
        'type': 'cifar',
        'name': 'wrn',
        'init_mode': 'kaiming_fan_out',
        'wrn':{
            'depth': 28,
            'initial_channels': 16,
            'widening_factor': 10,
            'drop_rate': 0.0
        },
        'normalize_layer': True
    }

    # Modify the test section for ResNet
    config['test']['checkpoint'] = "experiments/cifar10/wrn/exp00/checkpoint_00200.pth"
    config['test']['output_dir'] = config['test']['output_dir'].replace('vgg', 'wrn')

    # Ensure the directory for the new file path exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Save the modified config to a new file
    with open(new_file_path, 'w') as file:
        yaml.dump(config, file)

def create_wrn_configs_from_vgg(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') and 'vgg' in file:
                original_file_path = os.path.join(subdir, file)
                # Generate a new file path for ResNet configuration
                new_file_path = original_file_path.replace('vgg', 'wrn')
                modify_and_save_new_config_wrn(original_file_path, new_file_path)
                print(f'Original: {original_file_path} -> New: {new_file_path}')


def create_AT_resnet_configs_from_vgg(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') and 'vgg' in file:
                original_file_path = os.path.join(subdir, file)
                # Generate a new file path for ResNet configuration
                new_file_path = original_file_path.replace('cifar10', 'cifar10_AT').replace('vgg', 'resnet')
                AT_modify_and_save_new_resnet_config_vgg(original_file_path, new_file_path)
                print(f'Original: {original_file_path} -> New: {new_file_path}')

def AT_modify_and_save_new_resnet_config_vgg(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config['attack']['p']=='linf':
        config['attack']['epsilon']=25.5
        config['test']['checkpoint']="experiments/AT_cifar10/resnet_linf/exp00/checkpoint_00160.pth"
    elif config['attack']['p']=='l2':
        config['attack']['epsilon'] = 1275.0
        config['test']['checkpoint']="experiments/AT_cifar10/resnet_l2/exp00/checkpoint_00160.pth"
    else:
        config['test']['checkpoint']="experiments/AT_cifar10/resnet_linf/exp00/checkpoint_00160.pth"

    config['defense']['blacklight']=False

    config['test']['output_dir'] = config['test']['output_dir'].replace('cifar10', 'cifar10_AT').replace('vgg', 'resnet')
    config['test']['batch_size'] = 1

    # Modify the model section for ResNet
    config['model'] = {
        'type': 'cifar',
        'name': 'resnet',
        'init_mode': 'kaiming_fan_out',
        'resnet': {
            'depth': 110,
            'initial_channels': 16,
            'block_type': 'basic'
        },
        'normalize_layer': True
    }



    # Ensure the directory for the new file path exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Save the modified config to a new file
    with open(new_file_path, 'w') as file:
        yaml.dump(config, file)

# root_dir = './cifar10'  # Change this to the root directory of your config files
# create_resnet_configs_from_vgg(root_dir)
# create_resnext_configs_from_vgg(root_dir)
# create_wrn_configs_from_vgg(root_dir)

# root_dir = './cifar10'  # Change this to the root directory of your config files
# blacklight_path='./cifar10_blacklight'
# create_blacklight_configs_from_vgg(root_dir)
# create_resnet_configs_from_vgg(blacklight_path)
# create_resnext_configs_from_vgg(blacklight_path)
# create_wrn_configs_from_vgg(blacklight_path)

# root_dir = './cifar10'  # Change this to the root directory of your config files
# RAND_path='./cifar10_RAND'
# create_RAND_configs_from_vgg(root_dir)
# create_resnet_configs_from_vgg(RAND_path)
# create_resnext_configs_from_vgg(RAND_path)
# create_wrn_configs_from_vgg(RAND_path)

# root_dir = './cifar10'  # Change this to the root directory of your config files
# post_RAND_path='./cifar10_post_RAND'
# create_post_RAND_configs_from_vgg(root_dir)
# create_resnet_configs_from_vgg(post_RAND_path)
# create_resnext_configs_from_vgg(post_RAND_path)
# create_wrn_configs_from_vgg(post_RAND_path)

root_dir = './cifar10'
create_AT_resnet_configs_from_vgg(root_dir)