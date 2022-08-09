import pprint
import wandb
import sys  
import os

from train import train

sweep_config = {
    'method': 'grid'
    }

parameters_dict = {
    'num_labeled': {
        'values': [0, 100, 1000]
    },
    # 'alpha': {
    #     'values': [0, 0.01, 1, 10, 100]
    # },
    'beta': {
        'values': [0, 0.0001, 0.001, 1]
    },
    'delta': {
        'values': [1, 10, 100]
    },
}

parameters_dict.update({
    'flatten': {'value': False},
    'labeled_percentage': {'value': None},
    # 'num_labeled': {'value': 100},
    'max_epochs': {'value': 100},
    'alpha': {'value': 100},
    'beta': {'value': 0},
    'zeta': {'value': 1},
    'eta': {'value': 0.5},
    # 'delta': {'value': 1},
    'enc_out_dim': {'value': 1024},
    'latent_dim': {'value': 1024},
    }
)


sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="un2full vib")

print(sweep_id)

# wandb.agent(os.getenv("sweep_id"), train)