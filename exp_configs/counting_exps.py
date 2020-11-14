from haven import haven_utils as hu
import itertools, copy
RUNS = [0,1,2]
EXP_GROUPS = {}

splits = []

for s in ['random_wright', 'random_amazon', 'random_all']:
     splits.append({'name':'cows_json',
                          'sampler':'balanced',
                          'train_split_number': 0,
                          'val_split_number': 0,
                          'stratification': s})

splits2 = []

for s in [
          # 'random_wright', 
          'random_amazon',
          #  'random_all'
          ]:
     splits2.append({'name':'cows_json',
                          'sampler':'balanced',
                          'train_split_number': 0,
                          'val_split_number': 0,
                          'test_split_number': 0,
                          'stratification': s})


EXP_GROUPS['cows_counting'] = hu.cartesian_exp_group({
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': splits2,
                        'dataset_size':[
                                        {'train':'all', 'val':'all'},
                                        ],
                        # 'dataset_size':
                        'runs':RUNS,
                        'max_epoch': [500],
                        'optimizer': [ "adam"], 
                        'lr': [1e-5,],
                        'model':[
                              #   {'name':'semseg', 'loss':'lcfcn_consistency',
                              #               'base':'fcn8_vgg16',
                              #               'n_channels':3, 'n_classes':1},
                                            
                              {'name':'semseg', 'loss':'lcfcn_nopretrain',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':1},
                                            
                           
                              {'name':'semseg', 'loss':'density',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':1},
                                {'name':'semseg', 'loss':'lcfcn',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':1},
                                 
                                            ]
                        })