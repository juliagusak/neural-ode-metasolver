import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


import numpy as np
from collections import defaultdict

from MegaAdversarial.src.utils.runner import  test, test_ensemble, fix_seeds
from MegaAdversarial.src.attacks import (
    Clean,
    FGSM,
    PGD,
    LabelSmoothing,
    AdversarialVertex,
    AdversarialVertexExtra,
    FeatureScatter,
    NCEScatter,
    NCEScatterWithBuffer,
    MetaAttack,
    FGSM2Ensemble
)



def run_attack(model, epsilons, attack_modes, loaders, device='cuda', solvers_kwargs = None,
               at_lr = None, at_n_iter = None):
    robust_accuracies = defaultdict(list)
    
    for mode in attack_modes:
        for epsilon in epsilons:
#             CONFIG_PGD_TEST = {"eps": epsilon, "lr": 2.0 / 255 * 10, "n_iter": 20}
            CONFIG_PGD_TEST = {"eps": epsilon, "lr": at_lr, "n_iter": at_n_iter}
            CONFIG_FGSM_TEST = {"eps": epsilon}

            if mode == "clean":
                test_attack = Clean(model)
            elif mode == "fgsm":
                test_attack = FGSM(model, **CONFIG_FGSM_TEST)

            elif mode == "at":
                test_attack = PGD(model, **CONFIG_PGD_TEST)

#             elif mode == "at_ls":
#                 test_attack = PGD(model, **CONFIG_PGD_TEST) # wrong func, fix this

#             elif mode == "av":
#                 test_attack = PGD(model, **CONFIG_PGD_TEST) # wrong func, fix this

#             elif mode == "fs":
#                 test_attack = PGD(model, **CONFIG_PGD_TEST) # wrong func, fix this

            print("Attack {}".format(mode))    
            test_metrics = test(loaders["val"], model, test_attack, device, show_progress=True, solvers_kwargs = solvers_kwargs)
            test_log = f"Test: | " + " | ".join(
                map(lambda x: f"{x[0]}: {x[1]:.6f}", test_metrics.items())
            )
            print(test_log)
            
            robust_accuracies['accuracy_{}'.format(mode)].append(test_metrics['accuracy_adv'])
      
    return robust_accuracies



def run_attack2ensemble(models, epsilons, attack_modes, loaders, device='cuda', solvers_kwargs_arr = None,
               at_lr = None, at_n_iter = None):
    robust_accuracies = defaultdict(list)
    
    for mode in attack_modes:
        for epsilon in epsilons:
#             CONFIG_PGD_TEST = {"eps": epsilon, "lr": 2.0 / 255 * 10, "n_iter": 20}
            CONFIG_PGD_TEST = {"eps": epsilon, "lr": at_lr, "n_iter": at_n_iter}
            CONFIG_FGSM_TEST = {"eps": epsilon}

            if mode == "fgsm":
                test_attack2ensemble = FGSM2Ensemble(models, **CONFIG_FGSM_TEST)
            else: 
                raise NotImplementedError

            print("Attack {}".format(mode))    
            test_metrics = test_ensemble(loaders["val"],
                                         models,
                                         test_attack2ensemble,
                                         device,
                                         show_progress=True,
                                         solvers_kwargs_arr = solvers_kwargs_arr)
            
            test_log = f"Test: | " + " | ".join(
                map(lambda x: f"{x[0]}: {x[1]:.6f}", test_metrics.items())
            )
            print(test_log)
            
            robust_accuracies['accuracy_{}'.format(mode)].append(test_metrics['accuracy_adv'])
      
    return robust_accuracies