#General imports
import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import platform
import random
import pandas as pd
import argparse
import json
import pprint
import yaml
import torch
import torch.multiprocessing as mp


##Torch imports
import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

##Matdeeplearn imports
from matdeeplearn import models, process, training
from matdeeplearn.models.utils import model_summary

import ray
from ray import tune



config_path = 'config.yml'
#os.path.exists(config_path)
# os
os.path.abspath(os.getcwd())

assert os.path.exists(config_path), (
    "Config file not found in " + config_path
  )
with open(config_path, "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)
config["Job"] = config["Job"]['Inductive_Conformal']
config["Models"] = config["Models"].get("MEGNet_demo")
world_size = torch.cuda.device_count()
print(world_size)
config["Processing"]["data_path"] = "data/pt_data/pt_data_2"

config["Job"]['model_path'] = 'IC_model.pth'

rank = 'cuda'
print(world_size)
data_path = config["Processing"]["data_path"]
job_parameters= config["Job"]
training_parameters= config["Training"]
model_parameters= config["Models"]
processing_args= config['Processing']

##DDP
training.ddp_setup(rank, world_size)
##some issues with DDP learning rate
if rank not in ("cpu", "cuda"):
    model_parameters["lr"] = model_parameters["lr"] * world_size

##Get dataset
dataset = process.get_dataset(data_path, training_parameters["target_index"], True,  processing_args= config['Processing'])

print('Done Processing')

if rank not in ("cpu", "cuda"):
    dist.barrier()

##Set up loader
(
    train_loader,
    val_loader,
    test_loader,
    train_sampler,
    train_dataset,
    val_dataset,
    test_dataset,
) = training.loader_setup(
    training_parameters["train_ratio"],
    training_parameters["val_ratio"],
    training_parameters["test_ratio"],
    model_parameters["batch_size"],
    dataset,
    rank,
    job_parameters["seed"],
    world_size,
)

##Set up model
model =training.model_setup(
    rank,
    model_parameters["model"],
    model_parameters,
    dataset,
    job_parameters["load_model"],
    job_parameters["model_path"],
    model_parameters.get("print_model", True),
)

##Set-up optimizer & scheduler
optimizer = getattr(torch.optim, model_parameters["optimizer"])(
    model.parameters(),
    lr=model_parameters["lr"],
    **model_parameters["optimizer_args"]
)
scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
    optimizer, **model_parameters["scheduler_args"]
)

##Start training
model = training.trainer(
    rank,
    world_size,
    model,
    optimizer,
    scheduler,
    training_parameters["loss"],
    train_loader,
    val_loader,
    train_sampler,
    model_parameters["epochs"],
    training_parameters["verbosity"],
    "my_model_temp.pth",
)

if rank in (0, "cpu", "cuda"):

    train_error = val_error = test_error = float("NaN")

    ##workaround to get training output in DDP mode
    ##outputs are slightly different, could be due to dropout or batchnorm?
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_parameters["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = training.DataLoader(
                    test_dataset,
                    batch_size=model_parameters["batch_size"],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )

    val_loader = training.DataLoader(
                    val_dataset,
                    batch_size=model_parameters["batch_size"],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )

    ##Get train error in eval mode
    train_error, train_out = training.evaluate(
        train_loader, model, training_parameters["loss"], rank, out=True
    )
    print("Train Error: {:.5f}".format(train_error))

    ##Get val error
    if val_loader != None:
        val_error, val_out = training.evaluate(
            val_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Val Error: {:.5f}".format(val_error))

    ##Get test error
    if test_loader != None:
        test_error, test_out = training.evaluate(
            test_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Test Error: {:.5f}".format(test_error))
        
        
if job_parameters["save_model"] == "True":

    if rank not in ("cpu", "cuda"):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "full_model": model,
            },
            job_parameters["model_path"],
        )
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "full_model": model,
            },
            job_parameters["model_path"],
        )
target_train = pd.DataFrame(train_out, columns=['index', 'target', 'predicted'])
target_val = pd.DataFrame(val_out, columns=['index', 'target', 'predicted']) 
target_test = pd.DataFrame(test_out, columns=['index', 'target', 'predicted']) 
target_errors = pd.concat([target_train,target_val,target_test], axis = 0)
target_errors = target_errors.sort_values(list(target_errors), ascending=True)
target_errors['error'] = np.absolute(target_errors['target'].apply(float) - target_errors['predicted'].apply(float))

indices_list = target_errors['index'].to_list()
my_ind = pd.read_csv(os.path.join(os.getcwd(),data_path,'targets.csv'),header = None )

target_errors = target_errors.sort_values('index')

target_errors['index'] = target_errors['index'].apply(int)
indices_list = target_errors['index'].to_list()

all_ind = my_ind[0].to_list()
main_list = list(set(all_ind) - set(indices_list))

#print(main_list)
my_index = main_list[0]

new_df = pd.DataFrame({"index":my_index,"target":0, 'predicted':0, "error":0}, index=[19801])
target_errors = target_errors.append(new_df)
target_errors = target_errors.reset_index(drop=True)
target_errors = target_errors.sort_values('index')

target_errors = target_errors.sort_values('index')
target_errors = target_errors.reset_index(drop=True)
target_errors[['index','error']].to_csv(os.path.join(os.getcwd(),data_path,'error_targets.csv'), index = False, header=False)

new_data = process.get_dataset_error(data_path ,training_parameters["target_index"], False, processing_args)


error_train_subset =  torch.utils.data.Subset(new_data, train_dataset.indices)
error_val_subset = torch.utils.data.Subset(new_data, val_dataset.indices)
error_test_subset = torch.utils.data.Subset(new_data, test_dataset.indices)

#Checking whether tthe original training and new training datasets match
print("Checking the training errors are aligned:")
t_i = error_train_subset.indices[1]
print(target_errors.iloc[t_i])
print(new_data[t_i])
print(train_dataset[1])

train_loader_e = DataLoader(
    error_train_subset,
    batch_size=model_parameters["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

val_loader_e = DataLoader(
                error_val_subset,
                batch_size=model_parameters["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

test_loader_e = DataLoader(
                error_test_subset,
                batch_size=model_parameters["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

model_errors = training.model_setup(
        rank,
        model_parameters["model"],
        model_parameters,
        new_data,
        job_parameters["load_model"],
        job_parameters["model_path"],
        model_parameters.get("print_model", True),
    ) 

optimizer = getattr(torch.optim, model_parameters["optimizer"])(
    model_errors.parameters(),
    lr=model_parameters["lr"],
    **model_parameters["optimizer_args"]
)
scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
    optimizer, **model_parameters["scheduler_args"]
)

##Start training
model_errors = training.trainer(
    rank,
    world_size,
    model_errors,
    optimizer,
    scheduler,
    training_parameters["loss"],
    train_loader_e,
    val_loader_e,
    train_sampler,
    model_parameters["epochs"],
    training_parameters["verbosity"],
    "my_model_error_temp.pth",
)

train_error_e, train_out_e = training.evaluate(
    train_loader_e, model_errors, training_parameters["loss"], rank, out=True)

val_error_e, val_out_e = training.evaluate(
    val_loader_e, model_errors, training_parameters["loss"], rank, out=True)

test_error_e, test_out_e = training.evaluate(
    test_loader_e, model_errors, training_parameters["loss"], rank, out=True)


target_train_e = pd.DataFrame(train_out_e, columns=['index', 'target_error', 'predicted_error'])
target_val_e = pd.DataFrame(val_out_e, columns=['index', 'target_error', 'predicted_error']) 
target_test_e = pd.DataFrame(test_out_e, columns=['index', 'target_error', 'predicted_error']) 
target_errors_e = pd.concat([target_train_e,target_val_e,target_test_e], axis = 0)
target_errors_e = target_errors_e.sort_values(list(target_errors_e), ascending=True)
target_errors_e['error_2'] = np.absolute(target_errors_e['target_error'].apply(float) - target_errors_e['predicted_error'].apply(float))

target_val_e_2 = copy.copy(target_val_e)
target_val_e_2['target_error'] = target_val_e_2['target_error'].apply(float)
target_val_e_2['predicted_error'] = target_val_e_2['predicted_error'].apply(float)
target_val_e_2['alpha'] = np.abs(target_val_e_2['target_error']/target_val_e_2['predicted_error'])

target_val_e_2 = target_val_e_2.sort_values(['alpha'], axis=0, ascending=True)
alpha = np.percentile(target_val_e_2['alpha'], 95)

target_train_e['dataset'] = "train"
target_val_e['dataset'] = "val"
target_test_e['dataset'] = "test"
overall_e = pd.concat([target_train_e, target_val_e, target_test_e])
overall_e['index'] = overall_e['index'].apply(int)
combined_errors = target_errors.merge(overall_e, on='index', how='left')
combined_errors['predicted_error'] = combined_errors['predicted_error'].apply(float)
combined_errors['noncomformity']  = alpha
combined_errors['alpha_val'] = combined_errors['predicted_error']*alpha
combined_errors['sd'] = combined_errors['alpha_val']/2
combined_errors.to_csv(os.path.join(os.getcwd(),'Total_Errors_Confidence_Interval_Conformal.csv'), index = False, header=True)