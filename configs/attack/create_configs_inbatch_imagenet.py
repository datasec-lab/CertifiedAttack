import os
import yaml


def modify_and_save_vgg_config_post_RAND(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Modify the model section for ResNet

    config['dataset']['name']="ImageNet"
    config['dataset']['image_size'] = 224
    config['dataset']['n_channels'] = 3
    config['dataset']['n_classes'] = 1000


    config['model'] = {
        'type': 'imagenet',
        'name': 'vgg',
        'init_mode': 'kaiming_fan_out',
        'vgg': {
            'n_channels': [64, 128, 256, 512, 512],
            'n_layers': [2, 2, 3, 3, 3],
            'use_bn': True
        },
        'normalize_layer': True
    }

    # Modify the test section for ResNet
    config['test']['checkpoint'] = 'experiments/imagenet/vgg16/exp00/checkpoint_00090.pth'
    config['test']['output_dir'] = config['test']['output_dir'].replace('cifar10', 'imagenet_post_RAND')


    if config['attack']['p']=='linf':
        config['attack']['epsilon']=25.5
    if config['attack']['p']=='l2':
        config['attack']['epsilon'] = 10200.0
    config['attack']['max_loss_queries']=1000

    config['defense']['blacklight']=False
    config['defense']['post_sigma']=0.2

    config['test']['batch_size'] = 1
    # Ensure the directory for the new file path exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Save the modified config to a new file
    with open(new_file_path, 'w') as file:
        yaml.dump(config, file)


def modify_vgg_configs_post_RAND(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') and 'vgg' in file:
                original_file_path = os.path.join(subdir, file)
                # Generate a new file path for ResNet configuration
                new_file_path = original_file_path.replace('cifar10', 'imagenet_post_RAND')
                modify_and_save_vgg_config_post_RAND(original_file_path, new_file_path)
                print(f'Original: {original_file_path} -> New: {new_file_path}')

def modify_and_save_vgg_config_RAND(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Modify the model section for ResNet

    config['dataset']['name']="ImageNet"
    config['dataset']['image_size'] = 224
    config['dataset']['n_channels'] = 3
    config['dataset']['n_classes'] = 1000


    config['model'] = {
        'type': 'imagenet',
        'name': 'vgg',
        'init_mode': 'kaiming_fan_out',
        'vgg': {
            'n_channels': [64, 128, 256, 512, 512],
            'n_layers': [2, 2, 3, 3, 3],
            'use_bn': True
        },
        'normalize_layer': True
    }

    # Modify the test section
    config['test']['checkpoint'] = 'experiments/imagenet/vgg16/exp00/checkpoint_00090.pth'
    config['test']['output_dir'] = config['test']['output_dir'].replace('cifar10', 'imagenet_RAND')


    if config['attack']['p']=='linf':
        config['attack']['epsilon']=25.5
    if config['attack']['p']=='l2':
        config['attack']['epsilon'] = 10200.0
    config['attack']['max_loss_queries']=1000

    config['defense']['blacklight']=False
    config['defense']['sigma']=0.02

    config['test']['batch_size'] = 1
    # Ensure the directory for the new file path exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Save the modified config to a new file
    with open(new_file_path, 'w') as file:
        yaml.dump(config, file)


def modify_vgg_configs_RAND(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') and 'vgg' in file:
                original_file_path = os.path.join(subdir, file)
                # Generate a new file path for ResNet configuration
                new_file_path = original_file_path.replace('cifar10', 'imagenet_RAND')
                modify_and_save_vgg_config_RAND(original_file_path, new_file_path)
                print(f'Original: {original_file_path} -> New: {new_file_path}')
def modify_and_save_vgg_config_blacklight(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Modify the model section for ResNet

    config['dataset']['name']="ImageNet"
    config['dataset']['image_size'] = 224
    config['dataset']['n_channels'] = 3
    config['dataset']['n_classes'] = 1000

    config['model'] = {
        'type': 'imagenet',
        'name': 'vgg',
        'init_mode': 'kaiming_fan_out',
        'vgg': {
            'n_channels': [64, 128, 256, 512, 512],
            'n_layers': [2, 2, 3, 3, 3],
            'use_bn': True
        },
        'normalize_layer': True
    }

    # Modify the test section for ResNet
    config['test']['checkpoint'] = 'experiments/imagenet/vgg16/exp00/checkpoint_00090.pth'
    config['test']['output_dir'] = config['test']['output_dir'].replace('cifar10', 'imagenet_blacklight')


    if config['attack']['p']=='linf':
        config['attack']['epsilon']=25.5
    if config['attack']['p']=='l2':
        config['attack']['epsilon'] = 10200.0
    config['attack']['max_loss_queries']=1000

    config['defense']['blacklight']=True

    config['test']['batch_size'] = 1
    # Ensure the directory for the new file path exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    # Save the modified config to a new file
    with open(new_file_path, 'w') as file:
        yaml.dump(config, file)

def modify_vgg_configs_blacklight(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') and 'vgg' in file:
                original_file_path = os.path.join(subdir, file)
                # Generate a new file path for ResNet configuration
                new_file_path = original_file_path.replace('cifar10', 'imagenet_blacklight')
                modify_and_save_vgg_config_blacklight(original_file_path, new_file_path)
                print(f'Original: {original_file_path} -> New: {new_file_path}')


def modify_and_save_new_config_resnet(file_path, new_file_path):
    with open(file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Modify the model section for ResNet
    config['model'] = {
        'type': 'imagenet',
        'name': 'resnet',
        'init_mode': 'kaiming_fan_out',
        'resnet': {
            'initial_channels': 64,
            'block_type': 'basic',
            'n_blocks':[2,2,2,2]
        },
        'normalize_layer': True
    }

    # Modify the test section for ResNet
    config['test']['checkpoint'] = "experiments/imagenet/resnet18/exp00/checkpoint_00090.pth"
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



root_dir = './cifar10'  # Change this to the root directory of your config files

modify_vgg_configs_blacklight(root_dir)
create_resnet_configs_from_vgg('./imagenet_blacklight')

modify_vgg_configs_RAND(root_dir)
create_resnet_configs_from_vgg('./imagenet_RAND')

modify_vgg_configs_post_RAND(root_dir)
create_resnet_configs_from_vgg('./imagenet_post_RAND')