import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import Namespace
from .metrics import accuracy
from sopa.src.solvers.utils import noise_params
from MegaAdversarial.src.attacks import (
    Clean,
    PGD,
    FGSM
)

CONFIG_PGD_TRAIN = {"eps": 0.3, "lr": 2.0 / 255, "n_iter": 7}
CONFIG_FGSM_TRAIN = {"eps": 0.3}

def train(itr,
          model,
          data_gen,
          solvers,
          solver_options,
          criterion,
          optimizer,
          batch_time_meter,
          f_nfe_meter,
          b_nfe_meter,
          device = 'cpu',
          dtype = torch.float32,
          is_odenet = True,
          args = None,
          logger = None,
          wandb_logger = None):
    
    end = time.time()

    optimizer.zero_grad()
    x, y = data_gen.__next__()
    x = x.to(device)
    y = y.to(device)
    
    ##### Noise params
    if args.noise_type is not None:
        for i in range(len(solvers)):
            solvers[i].u, solvers[i].v = noise_params(solvers[i].u0,
                                                      solvers[i].v0,
                                                      std = args.noise_sigma,
                                                      bernoulli_p = args.noise_prob,
                                                      noise_type = args.noise_type)
            solvers[i].build_ButcherTableau()

    if args.adv_training_mode == "clean":
        train_attack = Clean(model)
    elif args.adv_training_mode == "fgsm":
        train_attack = FGSM(model, **CONFIG_FGSM_TRAIN)
    elif args.adv_training_mode == "at":
        train_attack = PGD(model, **CONFIG_PGD_TRAIN)
    else:
        raise ValueError("Attack type not understood.")
    x, y = train_attack(x, y, {"solvers": solvers, "solver_options": solver_options})

    # Add noise:
    if args.data_noise_std > 1e-12:
        with torch.no_grad():
            x = x + args.data_noise_std * torch.randn_like(x)
    ##### Forward pass
    if is_odenet:
        logits = model(x, solvers, solver_options, Namespace(ss_loss=args.ss_loss))
    else:
        logits = model(x)

    xentropy = criterion(logits, y)
    if args.ss_loss:
        ss_loss = model.get_ss_loss()
        loss = xentropy + args.ss_loss_reg * ss_loss
    else:
        ss_loss = 0.
        loss = xentropy
    if wandb_logger is not None:
        wandb_logger.log({"xentropy": xentropy.item(),
                    "ss_loss": ss_loss,
                    "loss": loss.item(),
                    "log_func": "train"})
    # if logger is not None:
    #     fix

    ##### Compute NFE-forward
    if is_odenet:
        nfe_forward = 0
        for i in range(len(model.blocks)):
            nfe_forward += model.blocks[i].rhs_func.nfe
            model.blocks[i].rhs_func.nfe = 0

    loss.backward()
    optimizer.step()

    ##### Compute NFE-backward
    if is_odenet:
        nfe_backward = 0
        for i in range(len(model.blocks)):
            nfe_backward += model.blocks[i].rhs_func.nfe
            model.blocks[i].rhs_func.nfe = 0

    ##### Denoise params
    if args.noise_type is not None:
        for i in range(len(solvers)):
            solvers[i].u, solvers[i].v = solvers[i].u0, solvers[i].v0
            solvers[i].build_ButcherTableau()

    batch_time_meter.update(time.time() - end)
    if is_odenet:
        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
            
            
        
def validate_standalone(best_acc,
                        itr,
                        model,
                        train_eval_loader,
                        test_loader,
                        batches_per_epoch,
                        solvers,
                        solver_options,
                        batch_time_meter,
                        f_nfe_meter,
                        b_nfe_meter,
                        device = 'cpu',
                        dtype = torch.float32,
                        args = None,
                        logger = None,
                        wandb_logger=None):
    
    nsolvers = len(solvers)

    with torch.no_grad():
        train_acc = [0] * nsolvers
        val_acc = [0] * nsolvers

        for solver_id, solver in enumerate(solvers):

            train_acc_id = accuracy(model, train_eval_loader, device, [solver], solver_options)
            val_acc_id = accuracy(model, test_loader, device, [solver], solver_options)

            train_acc[solver_id] = train_acc_id
            val_acc[solver_id] = val_acc_id


            if val_acc_id > best_acc[solver_id]:
                best_acc[solver_id] = val_acc_id

                torch.save({'state_dict': model.state_dict(),
                            'args': args,
                            'solver_id':solver_id,
                            'val_solver_mode':solver_options.solver_mode,
                            'acc': val_acc_id},
                           os.path.join(os.path.join(args.save, str(args.timestamp)),
                                        'model_best_{}.pth'.format(solver_id)))
                if wandb_logger is not None:
                    wandb_logger.save(os.path.join(os.path.join(args.save, str(args.timestamp)),
                                        'model_best_{}.pth'.format(solver_id)))
                
            if logger is not None:
                logger.info("Epoch {:04d} | SolverMode {} | SolverId {} | "
                        "TrainAcc {:.10f} | TestAcc {:.10f} | BestAcc {:.10f}".format(
                            itr // batches_per_epoch, solver_options.solver_mode, solver_id,
                            train_acc_id, val_acc_id, best_acc[solver_id]))
            if wandb_logger is not None:
                wandb_logger.log({
                    "epoch": itr // batches_per_epoch,
                    "solver_mode": solver_options.solver_mode,
                    "solver_id": solver_id,
                    "train_acc": train_acc_id,
                    "test_acc": val_acc_id,
                    "best_acc": best_acc[solver_id],
                    "log_func": "validate_standalone"
                })
            for i in range(len(model.blocks)):
                model.blocks[i].rhs_func.nfe = 0
        
    return best_acc



def validate_ensemble_switch(best_acc,
                             itr,
                             model,
                             train_eval_loader,
                             test_loader,
                             batches_per_epoch,
                             solvers,
                             solver_options,
                             batch_time_meter,
                             f_nfe_meter,
                             b_nfe_meter,
                             device = 'cpu',
                             dtype = torch.float32,
                             args = None,
                             logger=None,
                             wandb_logger=None):
    
    nsolvers = len(solvers)

    with torch.no_grad():

        train_acc = accuracy(model, train_eval_loader, device, solvers, solver_options)
        val_acc = accuracy(model, test_loader, device, solvers, solver_options)

        if val_acc > best_acc:
            best_acc = val_acc

            torch.save({'state_dict': model.state_dict(),
                        'args': args,
                        'solver_id':None,
                        'val_solver_mode':solver_options.solver_mode,
                        'acc': val_acc},
                       os.path.join(os.path.join(args.save, str(args.timestamp)),
                                    'model_best.pth'))
            if wandb_logger is not None:
                wandb_logger.save(os.path.join(os.path.join(args.save, str(args.timestamp)),
                                               'model_best.pth'))



        if logger is not None:
            logger.info("Epoch {:04d} | SolverMode {} | SolverId {} | "
                        "TrainAcc {:.10f} | TestAcc {:.10f} | BestAcc {:.10f}".format(
                            itr // batches_per_epoch, solver_options.solver_mode, None,
                            train_acc, val_acc, best_acc))
        if wandb_logger is not None:
            wandb_logger.log({
                "epoch": itr // batches_per_epoch,
                "solver_mode": solver_options.solver_mode,
                "solver_id": None,
                "train_acc": train_acc,
                "test_acc": val_acc,
                "best_acc": best_acc,
                "log_func": "validate_ensemble_switch"
            })


        for i in range(len(model.blocks)):
            model.blocks[i].rhs_func.nfe = 0
        
    return best_acc



def validate(best_acc,
            itr,
            model,
            train_eval_loader,
            test_loader,
            batches_per_epoch,
            solvers,
            val_solver_modes,
            batch_time_meter,
            f_nfe_meter,
            b_nfe_meter,
            device = 'cpu',
            dtype = torch.float32,
            args = None,
            logger = None,
            wandb_logger=None):
    
    for solver_mode in val_solver_modes:
        
        if solver_mode == 'standalone':
            
            val_solver_options = Namespace(solver_mode = 'standalone')
            best_acc['standalone'] = validate_standalone(best_acc['standalone'],
                                                          itr,
                                                          model,
                                                          train_eval_loader,
                                                          test_loader,
                                                          batches_per_epoch,
                                                          solvers = solvers,
                                                          solver_options = val_solver_options,
                                                          batch_time_meter = batch_time_meter,
                                                          f_nfe_meter = f_nfe_meter,
                                                          b_nfe_meter = b_nfe_meter,
                                                          device = device,
                                                          dtype = dtype,
                                                          args = args,
                                                          logger = logger,
                                                          wandb_logger = wandb_logger)
        elif solver_mode == 'ensemble':

            val_solver_options =  Namespace(solver_mode = 'ensemble',
                                            ensemble_weights = args.ensemble_weights,
                                            ensemble_prob = args.ensemble_prob)
        
            best_acc['ensemble'] = validate_ensemble_switch(best_acc['ensemble'],
                                                              itr,
                                                              model,
                                                              train_eval_loader,
                                                              test_loader,
                                                              batches_per_epoch,
                                                              solvers = solvers,
                                                              solver_options = val_solver_options,
                                                              batch_time_meter = batch_time_meter,
                                                              f_nfe_meter = f_nfe_meter,
                                                              b_nfe_meter = b_nfe_meter,
                                                              device = device,
                                                              dtype = dtype,
                                                              args = args,
                                                              logger = logger,
                                                             wandb_logger = wandb_logger)
        elif solver_mode == 'switch':
            
            val_solver_options =  Namespace(solver_mode = 'switch', switch_probs = args.switch_probs)
        
            best_acc['switch'] = validate_ensemble_switch(best_acc['switch'],
                                                           itr,
                                                           model,
                                                           train_eval_loader,
                                                           test_loader,
                                                           batches_per_epoch,
                                                           solvers = solvers,
                                                           solver_options = val_solver_options,
                                                           batch_time_meter = batch_time_meter,
                                                           f_nfe_meter = f_nfe_meter,
                                                           b_nfe_meter = b_nfe_meter,
                                                           device = device,
                                                           dtype = dtype,
                                                           args = args,
                                                           logger = logger,
                                                           wandb_logger = wandb_logger)
    if logger is not None:
        logger.info("Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f}".format(
            itr // batches_per_epoch,
            batch_time_meter.val, batch_time_meter.avg,
            f_nfe_meter.avg, b_nfe_meter.avg))
    if wandb_logger is not None:
        wandb_logger.log({
            "epoch": itr // batches_per_epoch,
            "batch_time_val": batch_time_meter.val,
            "nfe": f_nfe_meter.avg,
            "nbe": b_nfe_meter.avg,
            "log_func": "validate"
        })
    return best_acc