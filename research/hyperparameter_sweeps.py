import pprint
import wandb
import sys  
sys.path.insert(0, '../')

from learning_to_score.train import train

sweep_config = {
    'method': 'grid'
    }

parameters_dict = {
    'gamma': {
        'values': [0,0.001,0.01,0.1,1,10,100]
    }
}



parameters_dict.update({
    'flatten': {'value': False},
    'labeled_percentage': {'value': None},
    'num_labeled': {'value': 100},
    'max_epochs': {'value': 100},
    'alpha': {'value': 100},
    'beta': {'value': 0},
    'zeta': {'value': 1},
    'eta': {'value': 0.5},
    'delta': {'value': 1},
    'enc_out_dim': {'value': 1024},
    'latent_dim': {'value': 1024},
    }
)


sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="un2full vib")

wandb.agent(sweep_id, train)