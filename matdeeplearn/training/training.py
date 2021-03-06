##General imports
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
from matdeeplearn import models
import matdeeplearn.process as process
import matdeeplearn.training as training
from matdeeplearn.models.utils import model_summary

################################################################################
#  Training functions
################################################################################

##Train step, runs model in train mode

def nig_nll(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(two_blambda) \
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)

    return nll


def nig_reg(y, gamma, v, alpha, beta):
    error = F.l1_loss(y, gamma, reduction="none")
    evi = 2 * v + alpha
    return error * evi


def evidential_regresssion_loss(y, pred, coeff=0.05):
    gamma = torch.flatten(pred[0])
    v = torch.flatten(pred[1])
    alpha = torch.flatten(pred[2])
    beta = torch.flatten(pred[3])

    loss_nll = nig_nll(y, gamma, v, alpha, beta)
    loss_reg = nig_reg(y, gamma, v, alpha, beta)
    return loss_nll.mean() + coeff * loss_reg.mean()

def ev_evaluate(loader, model, loss_method, rank, out=False, my_coeff = 0.5):
    model.eval()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        with torch.no_grad():
            output = model(data)
            
            loss =  evidential_regresssion_loss(data.y, output, my_coeff)
            
            
            loss_all += loss * torch.flatten(output[0]).size(0)
            
            if out == True:
                if count == 0:
                    ids = [item for sublist in data.structure_id for item in sublist]
                    ids = [item for sublist in ids for item in sublist]
                    mu = torch.flatten(output[0])
                    v = torch.flatten(output[1])
                    alpha = torch.flatten(output[2])
                    beta = torch.flatten(output[3])

                    mu = mu.detach().cpu().numpy()
                    v = v.detach().cpu().numpy()
                    alpha = alpha.detach().cpu().numpy()
                    beta = beta.detach().cpu().numpy()
                    var = np.sqrt(beta / (v * (alpha - 1)))
                    target = data.y.cpu().numpy()
                    predict =  np.column_stack((mu, v, alpha, beta, var))
                else:
                    ids_temp = [
                        item for sublist in data.structure_id for item in sublist
                    ]
                    ids_temp = [item for sublist in ids_temp for item in sublist]
                    ids = ids + ids_temp
                    
                    mu = torch.flatten(output[0])
                    v = torch.flatten(output[1])
                    alpha = torch.flatten(output[2])
                    beta = torch.flatten(output[3])

                    mu = mu.detach().cpu().numpy()
                    v = v.detach().cpu().numpy()
                    alpha = alpha.detach().cpu().numpy()
                    beta = beta.detach().cpu().numpy()
                    var = np.sqrt(beta / (v * (alpha - 1)))
                    
                    predict = np.concatenate(
                        (predict, np.column_stack((mu, v, alpha, beta, var))), axis=0
                    )
                    target = np.concatenate((target, data.y.cpu().numpy()), axis=0)
            
            count = count + torch.flatten(output[0]).size(0)
            
                
    loss_all = loss_all / count            
    if out == True:
        ids = [float(i) for i in ids]
        test_out = np.column_stack((ids, target, predict))
        mydf = pd.DataFrame(test_out)

        mydf.rename(columns={0: 'ids', 1: 'target', 2: 'mu', 3:'v', 4:'alpha', 5:'beta', 6:'var'}, inplace=True)
        mydf['SD'] = np.sqrt(mydf['var'])
        mydf['error'] = mydf['target'] - mydf['mu']
        
        return loss_all, mydf
    elif out == False:
        return loss_all


def train(model, optimizer, loader, loss_method, rank, my_coeff = 1):
    model.train()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        optimizer.zero_grad()
        output = model(data)
        # print(data.y.shape, output.shape)
        if loss_method == "evidential":
            loss =  evidential_regresssion_loss(data.y, output, my_coeff)
            loss.backward()
            loss_all += loss.detach() * torch.flatten(output[0]).size(0)
            optimizer.step()
            count = count + torch.flatten(output[0]).size(0)
        else:
            loss = getattr(F, loss_method)(output, data.y)
            loss.backward()
            loss_all += loss.detach() * output.size(0)
            optimizer.step()
            count = count + output.size(0)

        # clip = 10
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        

    loss_all = loss_all / count
    return loss_all


##Evaluation step, runs model in eval mode
def evaluate(loader, model, loss_method, rank, out=False, my_coeff = 0.5):
    if loss_method == "evidential":
        return ev_evaluate(loader, model, loss_method, rank, out, my_coeff)
    else:
        model.eval()
        loss_all = 0
        count = 0
        for data in loader:
            data = data.to(rank)
            with torch.no_grad():
                output = model(data)
                loss = getattr(F, loss_method)(output, data.y)
                loss_all += loss * output.size(0)

                if out == True:
                    if count == 0:
                        ids = [item for sublist in data.structure_id for item in sublist]
                        ids = [item for sublist in ids for item in sublist]
                        predict = output.data.cpu().numpy()
                        target = data.y.cpu().numpy()
                    else:
                        ids_temp = [
                            item for sublist in data.structure_id for item in sublist
                        ]
                        ids_temp = [item for sublist in ids_temp for item in sublist]
                        ids = ids + ids_temp
                        predict = np.concatenate(
                            (predict, output.data.cpu().numpy()), axis=0
                        )
                        target = np.concatenate((target, data.y.cpu().numpy()), axis=0)

                count = count + output.size(0)

        loss_all = loss_all / count

        if out == True:
            test_out = np.column_stack((ids, target, predict))
            return loss_all, test_out
        elif out == False:
            return loss_all


##Model trainer
def trainer(
    rank,
    world_size,
    model,
    optimizer,
    scheduler,
    loss,
    train_loader,
    val_loader,
    train_sampler,
    epochs,
    verbosity,
    filename = "my_model_temp.pth",
    coeff = 1,
):
    train_error = val_error = test_error = epoch_time = float("NaN")
    train_start = time.time()
    best_val_error = 1e10
    model_best = model
    ##Start training over epochs loop
    for epoch in range(1, epochs + 1):

        lr = scheduler.optimizer.param_groups[0]["lr"]
        if rank not in ("cpu", "cuda"):
            train_sampler.set_epoch(epoch)
        ##Train model
        train_error = train(model, optimizer, train_loader, loss, rank=rank, my_coeff = coeff)
        if rank not in ("cpu", "cuda"):
            torch.distributed.reduce(train_error, dst=0)
            train_error = train_error / world_size
        
        ##Get validation performance
        if rank not in ("cpu", "cuda"):
            dist.barrier()
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                val_error = evaluate(
                    val_loader, model.module, loss, rank=rank, out=False, my_coeff = coeff
                )
            else:
                val_error = evaluate(val_loader, model, loss, rank=rank, out=False, my_coeff = coeff)

        ##Train loop timings
        epoch_time = time.time() - train_start
        train_start = time.time()

        ##remember the best val error and save model and checkpoint        
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if val_error == float("NaN") or val_error < best_val_error:
                if rank not in ("cpu", "cuda"):
                    model_best = copy.deepcopy(model.module)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
                else:
                    model_best = copy.deepcopy(model)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
            best_val_error = min(val_error, best_val_error)
        elif val_loader == None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                model_best = copy.deepcopy(model.module)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )
            else:
                model_best = copy.deepcopy(model)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )

        ##scheduler on train error
        
        scheduler.step(train_error)

        ##Print performance
        if epoch % verbosity == 0:
            if rank in (0, "cpu", "cuda"):
                print(
                    "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                        epoch, lr, train_error, val_error, epoch_time
                    )
                )

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    return model_best


##Write results to csv file
def write_results(output, filename):
    shape = output.shape
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        for i in range(0, len(output)):
            if i == 0:
                csvwriter.writerow(
                    ["ids"]
                    + ["target"] * int((shape[1] - 1) / 2)
                    + ["prediction"] * int((shape[1] - 1) / 2)
                )
            elif i > 0:
                csvwriter.writerow(output[i - 1, :])


##Pytorch ddp setup
def ddp_setup(rank, world_size):
    if rank in ("cpu", "cuda"):
        return
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if platform.system() == 'Windows':
        dist.init_process_group("gloo", rank=rank, world_size=world_size)    
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True


##Pytorch model setup
def model_setup(
    rank,
    model_name,
    model_params,
    dataset,
    load_model=False,
    model_path=None,
    print_model=True,
):
    model = getattr(models, model_name)(
        data=dataset, **(model_params if model_params is not None else {})
    ).to(rank)
    if load_model == "True":
        assert os.path.exists(model_path), "Saved model not found"
        if str(rank) in ("cpu"):
            saved = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            saved = torch.load(model_path)
        model.load_state_dict(saved["model_state_dict"])
        # optimizer.load_state_dict(saved['optimizer_state_dict'])

    # DDP
    if rank not in ("cpu", "cuda"):
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        # model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    if print_model == True and rank in (0, "cpu", "cuda"):
        model_summary(model)
    return model


##Pytorch loader setup
def loader_setup(
    train_ratio,
    val_ratio,
    test_ratio,
    batch_size,
    dataset,
    rank,
    seed,
    world_size=0,
    num_workers=0,
):
    ##Split datasets
    train_dataset, val_dataset, test_dataset = process.split_data(
        dataset, train_ratio, val_ratio, test_ratio, seed
    )

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

    ##Load data
    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    # may scale down batch size if memory is an issue
    if rank in (0, "cpu", "cuda"):
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        val_dataset,
        test_dataset,
    )


def loader_setup_CV(index, batch_size, dataset, rank, world_size=0, num_workers=0):
    ##Split datasets
    train_dataset = [x for i, x in enumerate(dataset) if i != index]
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = dataset[index]

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    if rank in (0, "cpu", "cuda"):
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, test_loader, train_sampler, train_dataset, test_dataset


################################################################################
#  Trainers
################################################################################

###Regular training with train, val, test split
def train_regular(
    rank,
    world_size,
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
    processing_args = None
):
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size

    ##Get dataset
    dataset = process.get_dataset(data_path, training_parameters["target_index"], False, processing_args)

    if rank not in ("cpu", "cuda"):
        dist.barrier()
    if model_parameters['coeff'] is not None:
        my_coeff = model_parameters['coeff'] 
    ##Set up loader
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = loader_setup(
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
    model = model_setup(
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
    model = trainer(
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
        my_coeff
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
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_parameters["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=model_parameters["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        if training_parameters["loss"] == "evidential":
            
            ##Get train error in eval mode
            train_error, train_out = ev_evaluate(
                train_loader, model, training_parameters["loss"], rank, True, my_coeff
            )
            print("Train Error: {:.5f}".format(train_error))

            ##Get val error
            if val_loader != None:
                val_error, val_out = ev_evaluate(
                    val_loader, model, training_parameters["loss"], rank, True, my_coeff
                )
                print("Val Error: {:.5f}".format(val_error))

            ##Get test error
            if test_loader != None:
                test_error, test_out = ev_evaluate(
                    test_loader, model, training_parameters["loss"], rank, True, my_coeff
                )
                print("Test Error: {:.5f}".format(test_error))
        else:
            
            ##Get train error in eval mode
            train_error, train_out = evaluate(
                train_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Train Error: {:.5f}".format(train_error))

            ##Get val error
            if val_loader != None:
                val_error, val_out = evaluate(
                    val_loader, model, training_parameters["loss"], rank, out=True
                )
                print("Val Error: {:.5f}".format(val_error))

            ##Get test error
            if test_loader != None:
                test_error, test_out = evaluate(
                    test_loader, model, training_parameters["loss"], rank, out=True
                )
                print("Test Error: {:.5f}".format(test_error))

        ##Save model
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

        ##Write outputs
        if job_parameters["write_output"] == "True":
            if training_parameters["loss"] == "evidential":
                train_name = str(job_parameters["job_name"]) + "_train_outputs.csv"
                train_out.to_csv(path_or_buf = train_name, index = False)
                
                if val_loader != None:
                    val_name = str(job_parameters["job_name"]) + "_val_outputs.csv"
                    val_out.to_csv(path_or_buf = val_name, index = False)
                 
                if test_loader != None:
                    test_name = str(job_parameters["job_name"]) + "_test_outputs.csv"
                    test_out.to_csv(path_or_buf = test_name, index = False)
            else:
                write_results(
                    train_out, str(job_parameters["job_name"]) + "_train_outputs.csv"
                )
                if val_loader != None:
                    write_results(
                        val_out, str(job_parameters["job_name"]) + "_val_outputs.csv"
                    )
                if test_loader != None:
                    write_results(
                        test_out, str(job_parameters["job_name"]) + "_test_outputs.csv"
                    )

        if rank not in ("cpu", "cuda"):
            dist.destroy_process_group()

        ##Write out model performance to file
        error_values = np.array((train_error.cpu(), val_error.cpu(), test_error.cpu()))
        if job_parameters.get("write_error") == "True":
            np.savetxt(
                job_parameters["job_name"] + "_errorvalues.csv",
                error_values[np.newaxis, ...],
                delimiter=",",
            )

        return error_values


###Predict using a saved movel
def predict(dataset, loss, job_parameters=None):

    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##Loads predict dataset in one go, care needed for large datasets)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    ##Load saved model
    assert os.path.exists(job_parameters["model_path"]), "Saved model not found"
    if str(rank) == "cpu":
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cpu")
        )
    else:
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cuda")
        )
    model = saved["full_model"]
    model = model.to(rank)
    model_summary(model)

    ##Get predictions
    time_start = time.time()
    test_error, test_out = evaluate(loader, model, loss, rank, out=True)
    elapsed_time = time.time() - time_start

    print("Evaluation time (s): {:.5f}".format(elapsed_time))

    ##Write output
    if job_parameters["write_output"] == "True":
        write_results(
            test_out, str(job_parameters["job_name"]) + "_predicted_outputs.csv"
        )

    return test_error


###n-fold cross validation
def train_CV(
    rank,
    world_size,
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):

    job_parameters["load_model"] = "False"
    job_parameters["save_model"] = "False"
    job_parameters["model_path"] = None
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size

    ##Get dataset
    dataset = process.get_dataset(data_path, training_parameters["target_index"], False)

    ##Split datasets
    cv_dataset = process.split_data_CV(
        dataset, num_folds=job_parameters["cv_folds"], seed=job_parameters["seed"]
    )
    cv_error = 0

    for index in range(0, len(cv_dataset)):

        ##Set up model
        if index == 0:
            model = model_setup(
                rank,
                model_parameters["model"],
                model_parameters,
                dataset,
                job_parameters["load_model"],
                job_parameters["model_path"],
                print_model=True,
            )
        else:
            model = model_setup(
                rank,
                model_parameters["model"],
                model_parameters,
                dataset,
                job_parameters["load_model"],
                job_parameters["model_path"],
                print_model=False,
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

        ##Set up loader
        train_loader, test_loader, train_sampler, train_dataset, _ = loader_setup_CV(
            index, model_parameters["batch_size"], cv_dataset, rank, world_size
        )

        ##Start training
        model = trainer(
            rank,
            world_size,
            model,
            optimizer,
            scheduler,
            training_parameters["loss"],
            train_loader,
            None,
            train_sampler,
            model_parameters["epochs"],
            training_parameters["verbosity"],
            "my_model_temp.pth",
        )

        if rank not in ("cpu", "cuda"):
            dist.barrier()

        if rank in (0, "cpu", "cuda"):

            train_loader = DataLoader(
                train_dataset,
                batch_size=model_parameters["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            ##Get train error
            train_error, train_out = evaluate(
                train_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Train Error: {:.5f}".format(train_error))

            ##Get test error
            test_error, test_out = evaluate(
                test_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Test Error: {:.5f}".format(test_error))

            cv_error = cv_error + test_error

            if index == 0:
                total_rows = test_out
            else:
                total_rows = np.vstack((total_rows, test_out))

    ##Write output
    if rank in (0, "cpu", "cuda"):
        if job_parameters["write_output"] == "True":
            if test_loader != None:
                write_results(
                    total_rows, str(job_parameters["job_name"]) + "_CV_outputs.csv"
                )

        cv_error = cv_error / len(cv_dataset)
        print("CV Error: {:.5f}".format(cv_error))

    if rank not in ("cpu", "cuda"):
        dist.destroy_process_group()

    return cv_error


### Repeat training for n times
def train_repeat(
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):

    world_size = torch.cuda.device_count()
    job_name = job_parameters["job_name"]
    model_path = job_parameters["model_path"]
    job_parameters["write_error"] = "True"
    job_parameters["load_model"] = "False"
    job_parameters["save_model"] = "False"
    ##Loop over number of repeated trials
    for i in range(0, job_parameters["repeat_trials"]):

        ##new seed each time for different data split
        job_parameters["seed"] = np.random.randint(1, 1e6)

        if i == 0:
            model_parameters["print_model"] = True
        else:
            model_parameters["print_model"] = False

        job_parameters["job_name"] = job_name + str(i)
        job_parameters["model_path"] = str(i) + "_" + model_path

        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_regular(
                "cpu",
                world_size,
                data_path,
                job_parameters,
                training_parameters,
                model_parameters,
            )
        elif world_size > 0:
            if job_parameters["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(
                    training.train_regular,
                    args=(
                        world_size,
                        data_path,
                        job_parameters,
                        training_parameters,
                        model_parameters,
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if job_parameters["parallel"] == "False":
                print("Running on one GPU")
                training.train_regular(
                    "cuda",
                    world_size,
                    data_path,
                    job_parameters,
                    training_parameters,
                    model_parameters,
                )

    ##Compile error metrics from individual trials
    print("Individual training finished.")
    print("Compiling metrics from individual trials...")
    error_values = np.zeros((job_parameters["repeat_trials"], 3))
    for i in range(0, job_parameters["repeat_trials"]):
        filename = job_name + str(i) + "_errorvalues.csv"
        error_values[i] = np.genfromtxt(filename, delimiter=",")
    mean_values = [
        np.mean(error_values[:, 0]),
        np.mean(error_values[:, 1]),
        np.mean(error_values[:, 2]),
    ]
    std_values = [
        np.std(error_values[:, 0]),
        np.std(error_values[:, 1]),
        np.std(error_values[:, 2]),
    ]

    ##Print error
    print(
        "Training Error Avg: {:.3f}, Training Standard Dev: {:.3f}".format(
            mean_values[0], std_values[0]
        )
    )
    print(
        "Val Error Avg: {:.3f}, Val Standard Dev: {:.3f}".format(
            mean_values[1], std_values[1]
        )
    )
    print(
        "Test Error Avg: {:.3f}, Test Standard Dev: {:.3f}".format(
            mean_values[2], std_values[2]
        )
    )

    ##Write error metrics
    if job_parameters["write_output"] == "True":
        with open(job_name + "_all_errorvalues.csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                [
                    "",
                    "Training",
                    "Validation",
                    "Test",
                ]
            )
            for i in range(0, len(error_values)):
                csvwriter.writerow(
                    [
                        "Trial " + str(i),
                        error_values[i, 0],
                        error_values[i, 1],
                        error_values[i, 2],
                    ]
                )
            csvwriter.writerow(["Mean", mean_values[0], mean_values[1], mean_values[2]])
            csvwriter.writerow(["Std", std_values[0], std_values[1], std_values[2]])
    elif job_parameters["write_output"] == "False":
        for i in range(0, job_parameters["repeat_trials"]):
            filename = job_name + str(i) + "_errorvalues.csv"
            os.remove(filename)


###Hyperparameter optimization
# trainable function for ray tune (no parallel, max 1 GPU per job)
def tune_trainable(config, checkpoint_dir=None, data_path=None):

    # imports
    from ray import tune

    print("Hyperparameter trial start")
    hyper_args = config["hyper_args"]
    job_parameters = config["job_parameters"]
    processing_parameters = config["processing_parameters"]
    training_parameters = config["training_parameters"]
    model_parameters = config["model_parameters"]

    ##Merge hyperparameter parameters with constant parameters, with precedence over hyperparameter ones
    ##Omit training and job parameters as they should not be part of hyperparameter opt, in theory
    model_parameters = {**model_parameters, **hyper_args}
    processing_parameters = {**processing_parameters, **hyper_args}

    ##Assume 1 gpu or 1 cpu per trial, no functionality for parallel yet
    world_size = 1
    rank = "cpu"
    if torch.cuda.is_available():
        rank = "cuda"

    ##Reprocess data in a separate directory to prevent conflict
    if job_parameters["reprocess"] == "True":
        time = datetime.now()
        processing_parameters["processed_path"] = time.strftime("%H%M%S%f")
        processing_parameters["verbose"] = "False"
    data_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    data_path = os.path.join(data_path, processing_parameters["data_path"])
    data_path = os.path.normpath(data_path)
    print("Data path", data_path)

    ##Set up dataset
    dataset = process.get_dataset(
        data_path,
        training_parameters["target_index"],
        job_parameters["reprocess"],
        processing_parameters,
    )

    ##Set up loader
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        _,
        _,
    ) = loader_setup(
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
    model = model_setup(
        rank,
        model_parameters["model"],
        model_parameters,
        dataset,
        False,
        None,
        False,
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

    ##Load checkpoint
    if checkpoint_dir:
        model_state, optimizer_state, scheduler_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)

    ##Training loop
    for epoch in range(1, model_parameters["epochs"] + 1):
        lr = scheduler.optimizer.param_groups[0]["lr"]
        train_error = train(
            model, optimizer, train_loader, training_parameters["loss"], rank=rank
        )

        val_error = evaluate(
            val_loader, model, training_parameters["loss"], rank=rank, out=False
        )

        ##Delete processed data
        if epoch == model_parameters["epochs"]:
            if (
                job_parameters["reprocess"] == "True"
                and job_parameters["hyper_delete_processed"] == "True"
            ):
                shutil.rmtree(
                    os.path.join(data_path, processing_parameters["processed_path"])
                )
            print("Finished Training")

        ##Update to tune
        if epoch % job_parameters["hyper_iter"] == 0:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (
                        model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                    ),
                    path,
                )
            ##Somehow tune does not recognize value without *1
            tune.report(loss=val_error.cpu().numpy() * 1)
            # tune.report(loss=val_error)


# Tune setup
def tune_setup(
    hyper_args,
    job_parameters,
    processing_parameters,
    training_parameters,
    model_parameters,
):

    # imports
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune import CLIReporter

    ray.init()
    data_path = "_"
    local_dir = "ray_results"
    # currently no support for paralleization per trial
    gpus_per_trial = 1

    ##Set up search algo
    search_algo = HyperOptSearch(metric="loss", mode="min", n_initial_points=5)
    search_algo = ConcurrencyLimiter(
        search_algo, max_concurrent=job_parameters["hyper_concurrency"]
    )

    ##Resume run
    if os.path.exists(local_dir + "/" + job_parameters["job_name"]) and os.path.isdir(
        local_dir + "/" + job_parameters["job_name"]
    ):
        if job_parameters["hyper_resume"] == "False":
            resume = False
        elif job_parameters["hyper_resume"] == "True":
            resume = True
        # else:
        #    resume = "PROMPT"
    else:
        resume = False

    ##Print out hyperparameters
    parameter_columns = [
        element for element in hyper_args.keys() if element not in "global"
    ]
    parameter_columns = ["hyper_args"]
    reporter = CLIReporter(
        max_progress_rows=20, max_error_rows=5, parameter_columns=parameter_columns
    )

    ##Run tune
    tune_result = tune.run(
        partial(tune_trainable, data_path=data_path),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config={
            "hyper_args": hyper_args,
            "job_parameters": job_parameters,
            "processing_parameters": processing_parameters,
            "training_parameters": training_parameters,
            "model_parameters": model_parameters,
        },
        num_samples=job_parameters["hyper_trials"],
        # scheduler=scheduler,
        search_alg=search_algo,
        local_dir=local_dir,
        progress_reporter=reporter,
        verbose=job_parameters["hyper_verbosity"],
        resume=resume,
        log_to_file=True,
        name=job_parameters["job_name"],
        max_failures=4,
        raise_on_failed_trial=False,
        # keep_checkpoints_num=job_parameters["hyper_keep_checkpoints_num"],
        # checkpoint_score_attr="min-loss",
        stop={
            "training_iteration": model_parameters["epochs"]
            // job_parameters["hyper_iter"]
        },
    )

    ##Get best trial
    best_trial = tune_result.get_best_trial("loss", "min", "all")
    # best_trial = tune_result.get_best_trial("loss", "min", "last")

    return best_trial


###Simple ensemble using averages
def train_ensemble(
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):

    world_size = torch.cuda.device_count()
    job_name = job_parameters["job_name"]
    write_output = job_parameters["write_output"]
    model_path = job_parameters["model_path"]
    job_parameters["write_error"] = "True"
    job_parameters["write_output"] = "True"
    job_parameters["load_model"] = "False"
    ##Loop over number of repeated trials
    for i in range(0, len(job_parameters["ensemble_list"])):
        job_parameters["job_name"] = job_name + str(i)
        job_parameters["model_path"] = (
            str(i) + "_" + job_parameters["ensemble_list"][i] + "_" + model_path
        )

        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_regular(
                "cpu",
                world_size,
                data_path,
                job_parameters,
                training_parameters,
                model_parameters[job_parameters["ensemble_list"][i]],
            )
        elif world_size > 0:
            if job_parameters["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(
                    training.train_regular,
                    args=(
                        world_size,
                        data_path,
                        job_parameters,
                        training_parameters,
                        model_parameters[job_parameters["ensemble_list"][i]],
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if job_parameters["parallel"] == "False": 
                print("Running on one GPU")
                training.train_regular(
                    "cuda",
                    world_size,
                    data_path,
                    job_parameters,
                    training_parameters,
                    model_parameters[job_parameters["ensemble_list"][i]],
                )

    ##Compile error metrics from individual models
    print("Individual training finished.")
    print("Compiling metrics from individual models...")
    error_values = np.zeros((len(job_parameters["ensemble_list"]), 3))
    for i in range(0, len(job_parameters["ensemble_list"])):
        filename = job_name + str(i) + "_errorvalues.csv"
        error_values[i] = np.genfromtxt(filename, delimiter=",")
    mean_values = [
        np.mean(error_values[:, 0]),
        np.mean(error_values[:, 1]),
        np.mean(error_values[:, 2]),
    ]
    std_values = [
        np.std(error_values[:, 0]),
        np.std(error_values[:, 1]),
        np.std(error_values[:, 2]),
    ]

    # average ensembling, takes the mean of the predictions
    for i in range(0, len(job_parameters["ensemble_list"])):
        filename = job_name + str(i) + "_test_outputs.csv"
        test_out = np.genfromtxt(filename, delimiter=",", skip_header=1)
        if i == 0:
            test_total = test_out
        elif i > 0:
            test_total = np.column_stack((test_total, test_out[:, 2]))

    ensemble_test = np.mean(np.array(test_total[:, 2:]).astype(np.float), axis=1)
    ensemble_test_error = getattr(F, training_parameters["loss"])(
        torch.tensor(ensemble_test),
        torch.tensor(test_total[:, 1].astype(np.float)),
    )
    test_total = np.column_stack((test_total, ensemble_test))
    
    ##Print performance
    for i in range(0, len(job_parameters["ensemble_list"])):
        print(
            job_parameters["ensemble_list"][i]
            + " Test Error: {:.5f}".format(error_values[i, 2])
        )
    print(
        "Test Error Avg: {:.3f}, Test Standard Dev: {:.3f}".format(
            mean_values[2], std_values[2]
        )
    )
    print("Ensemble Error: {:.5f}".format(ensemble_test_error))
    
    ##Write output
    if write_output == "True" or write_output == "Partial":
        with open(
            str(job_name) + "_test_ensemble_outputs.csv", "w"
        ) as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(test_total) + 1):
                if i == 0:
                    csvwriter.writerow(
                        [
                            "ids",
                            "target",
                        ]
                        + job_parameters["ensemble_list"]
                        + ["ensemble"]
                    )
                elif i > 0:
                    csvwriter.writerow(test_total[i - 1, :])
    if write_output == "False" or write_output == "Partial":
        for i in range(0, len(job_parameters["ensemble_list"])):
            filename = job_name + str(i) + "_errorvalues.csv"
            os.remove(filename)
            filename = job_name + str(i) + "_test_outputs.csv"
            os.remove(filename)

##Obtains features from graph in a trained model and analysis with tsne
def analysis(
    dataset,
    model_path,
    tsne_args,
):

    # imports
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = []

    def hook(module, input, output):
        inputs.append(input)

    assert os.path.exists(model_path), "saved model not found"
    if str(rank) == "cpu":
        saved = torch.load(model_path, map_location=torch.device("cpu"))
    else:
        saved = torch.load(model_path, map_location=torch.device("cuda"))
    model = saved["full_model"]
    model_summary(model)

    print(dataset)

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()
    ##Grabs the input of the first linear layer after the GNN
    model.post_lin_list[0].register_forward_hook(hook)
    for data in loader:
        with torch.no_grad():
            data = data.to(rank)
            output = model(data)

    inputs = [i for sub in inputs for i in sub]
    inputs = torch.cat(inputs)
    inputs = inputs.cpu().numpy()
    print("Number of samples: ", inputs.shape[0])
    print("Number of features: ", inputs.shape[1])

    # only works for when targets has one index
    targets = dataset.data.y.numpy()

    # pca = PCA(n_components=2)
    # pca_out=pca.fit_transform(inputs)
    # print(pca_out.shape)
    # np.savetxt('pca.csv', pca_out, delimiter=',')
    # plt.scatter(pca_out[:,1],pca_out[:,0],c=targets,s=15)
    # plt.colorbar()
    # plt.show()
    # plt.clf()

    ##Start t-SNE analysis
    tsne = TSNE(**tsne_args)
    tsne_out = tsne.fit_transform(inputs)
    rows = zip(
        dataset.data.structure_id,
        list(dataset.data.y.numpy()),
        list(tsne_out[:, 0]),
        list(tsne_out[:, 1]),
    )

    with open("tsne_output.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        for row in rows:
            writer.writerow(row)

    fig, ax = plt.subplots()
    main = plt.scatter(tsne_out[:, 1], tsne_out[:, 0], c=targets, s=3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(main, ax=ax)
    stdev = np.std(targets)
    cbar.mappable.set_clim(
        np.mean(targets) - 2 * np.std(targets), np.mean(targets) + 2 * np.std(targets)
    )
    # cbar.ax.tick_params(labelsize=50)
    # cbar.ax.tick_params(size=40)
    plt.savefig("tsne_output.png", format="png", dpi=600)
    plt.show()


    ############### Functions for Ensemble CI Estimation #######################

#make sure to always include 0!
def create_random_indices(my_range, num_samples):
    indices = list()
    for samp in range(0, num_samples):
        cur_choices = random.choices(my_range, k = len(my_range))
        indices.append(cur_choices)
    return indices

#returns a list of training model splits while keeping a single val and test set
def split_data_bootstrap(
    dataset,
    train_ratio,
    val_ratio,
    test_ratio,
    seed=np.random.randint(1, 1e6),
    save=False,
    num_samples = 50,
):
    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <= 1:
        train_length = int(dataset_size * train_ratio)
        val_length = int(dataset_size * val_ratio)
        test_length = int(dataset_size * test_ratio)
        unused_length = dataset_size - train_length - val_length - test_length
        (
            train_dataset,
            val_dataset,
            test_dataset,
            unused_dataset,
        ) = torch.utils.data.random_split(
            dataset,
            [train_length, val_length, test_length, unused_length],
            generator=torch.Generator().manual_seed(seed),
        )
        
        
        total_ind = train_dataset.indices
        #c_range = range(0, len(total_ind))
        my_indices = create_random_indices(total_ind, num_samples)
        training_bootstrapped = []


        for ind in my_indices:
            my_tr = copy.copy(train_dataset)
            my_tr.indices = ind
            training_bootstrapped.append(my_tr)


        print(
            "train length:",
            train_length,
            "val length:",
            val_length,
            "test length:",
            test_length,
            "unused length:",
            unused_length,
            "seed :",
            seed,
        )
        return training_bootstrapped, val_dataset, test_dataset
    else:
        print("invalid ratios")
        
        
        
## Loading a list of bootstrapped training subsets in order to train ensemble models
def loader_setup_bootstrap(
    train_ratio,
    val_ratio,
    test_ratio,
    batch_size,
    dataset,
    rank,
    seed,
    world_size=0,
    num_workers=0,
    num_samples = 50,
):
    ##Split datasets
    train_sets, val_dataset, test_dataset = split_data_bootstrap(
        dataset, train_ratio, val_ratio, test_ratio, seed, num_samples = num_samples
    )
    
    #print(str(type(train_sets)) + "_1")

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

        #print(str(type(train_sets)) + "_2")
        
    ##Load data
    train_loader = val_loader = test_loader = None
    t_loader_list = []
    
    #print(type(train_sets))
    
    for train_dataset in train_sets:
        #print(str(type(train_sets)) + "_3")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )
        
        #print(str(type(t_loader_list)) + "_4")
        t_loader_list.append(copy.copy(train_loader))
        train_loader = None
        #print(str(type(train_sets)) + "_5")
    # may scale down batch size if memory is an issue
    if rank in (0, "cpu", "cuda"):
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            
    #print(str(type(train_sets)) + "_6")
    #print(str(type(copy.copy(train_sets))) + "****************")
    return(
        t_loader_list,
        val_loader,
        test_loader,
        train_sampler,
        train_sets,
        train_dataset,
        val_dataset,
        test_dataset
    )



#Training using the bootstrap method
def train_regular_bootstrap(
    rank,
    world_size,
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
    n_samples = 50,
    get_CI = True
):
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size

    ##Get dataset
    dataset = process.get_dataset(data_path, training_parameters["target_index"], False)

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    ##Set up loader
    (
        t_loaders,
        val_loader,
        test_loader,
        train_sampler,
        train_sets,
        train_dataset,
        _,
        _,
    ) = loader_setup_bootstrap(
        training_parameters["train_ratio"],
        training_parameters["val_ratio"],
        training_parameters["test_ratio"],
        model_parameters["batch_size"],
        dataset,
        rank,
        job_parameters["seed"],
        world_size,
        num_samples = n_samples,
    )
    
    if rank in (0, "cpu", "cuda"):
        train_error_tot = []
        val_error_tot = []
        test_error_tot = []

    for i in range(0, len(t_loaders)):
    ##Set up model
    
        my_split = job_parameters["model_n"].find('.')
       
        model_name = os.path.join(
            job_parameters["model_path"], 
            job_parameters["model_n"][:my_split] + "_" + 
            str(i) + job_parameters["model_n"][my_split:]
            )
        
        ########### USE .training #############
        model = training.model_setup(
            rank,
            model_parameters["model"],
            model_parameters,
            dataset,
            job_parameters["load_model"],
            model_name,
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
        ########### USE .training #############
        model = training.trainer(
            rank,
            world_size,
            model,
            optimizer,
            scheduler,
            training_parameters["loss"],
            t_loaders[i],
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
                train_sets[i],
                batch_size=model_parameters["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            ##Get train error in eval mode
            ########### USE .training #############
            train_error, train_out = evaluate(
                train_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Train Error: {:.5f}".format(train_error))

            ##Get val error
            if val_loader != None:
                ########### USE .training #############
                val_error, val_out = evaluate(
                    val_loader, model, training_parameters["loss"], rank, out=True
                )
                print("Val Error: {:.5f}".format(val_error))

            ##Get test error
            if test_loader != None:
                ########### USE .training #############
                test_error, test_out = evaluate(
                    test_loader, model, training_parameters["loss"], rank, out=True
                )
                print("Test Error: {:.5f}".format(test_error))

            ##Save model
            if job_parameters["save_model"] == "True":
                
                if rank not in ("cpu", "cuda"):
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                         model_name,
                    )
                else:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                         model_name,
                    )

            ##Write outputs
            model_name = os.path.join(
            job_parameters["model_path"], 
            job_parameters["model_n"][:my_split] + "_" + 
            str(i) + job_parameters["model_n"][my_split:]
            )

            if job_parameters["write_output"] == "True":
                ########### USE .training #############
                training.write_results(
                    train_out, os.path.join(job_parameters["model_path"], 
                    str(job_parameters["job_name"]) + "_" + str(i) + "_train_outputs.csv")
                )
                if val_loader != None:
                    ########### USE .training #############
                    training.write_results(
                        val_out, os.path.join(job_parameters["model_path"],
                        str(job_parameters["job_name"]) + "_" + str(i) + "_val_outputs.csv")
                    )
                if test_loader != None:
                    ########### USE .training #############
                    training.write_results(
                        test_out, os.path.join(job_parameters["model_path"],
                        str(job_parameters["job_name"]) + "_" + str(i) + "_test_outputs.csv")
                    )

            if rank not in ("cpu", "cuda"):
                dist.destroy_process_group()

        ##Write out model performance to file
            error_values = np.array((train_error.cpu(), val_error.cpu(), test_error.cpu()))
            if job_parameters.get("write_error") == "True":
                np.savetxt(
                    job_parameters["job_name"] + "_" + str(i) + "_errorvalues.csv",
                    error_values[np.newaxis, ...],
                    delimiter=",",
                )          
    if get_CI == True:
        count_ids(train_dataset, train_sets)
        predict_ci(training_parameters["loss"], n_samples, job_parameters, rank, val_loader, 'val')     
        predict_ci(training_parameters["loss"], n_samples, job_parameters, rank, test_loader, 'test')
    return error_values



#used to verify that all of our test values are equal
def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    val = []
    for x in iterator:
        val.append(np.array_equal(first, x))
    return all(val)


def predict_ci(loss, num_models, job_parameters, rank, loader, set_name, datset = None,  write_output = True):
    models = []
    predictions = []
    target_val = []
    labs_target = []
    my_split = job_parameters["model_n"].find('.')

    for i in range(0,num_models):
        if str(rank) == "cpu":
            saved = torch.load(
                os.path.join(job_parameters["model_path"],
                             job_parameters["model_n"][:my_split] + "_" + 
                             str(i) + job_parameters["model_n"][my_split:]),
                             map_location=torch.device("cpu")
            )
        else:
            saved = torch.load(
                os.path.join(job_parameters["model_path"],
                             job_parameters["model_n"][:my_split] + "_" + 
                             str(i) + job_parameters["model_n"][my_split:]),
                             map_location=torch.device("cuda")
            )

        model = saved["full_model"]
        model = model.to(rank)
        model_summary(model)
        time_start = time.time()
        test_error, test_out = evaluate(loader, model, loss, rank, out=True)
        elapsed_time = time.time() - time_start
        if i == 0:
            labs_target = test_out[:,[0,1]]
        target_val.append(test_out[:,1])
        predictions.append(test_out[:,2])

    print("All test values the same: " + str(all_equal(target_val)))
    float_pred = [i.astype(float) for i in predictions]

    means = np.mean(float_pred, axis = 0)
    my_sd = np.std(float_pred, axis = 0)
    two_sd = np.std(float_pred, axis = 0)*2
    top_ci = np.add(means, two_sd)
    bot_ci = np.subtract(means, two_sd)

    #labs_target = labs_target.reshape(labs_target.shape[0],-1)
    #targets = target_val[0].reshape(target_val[0].shape[0],-1)
    bot_ci = bot_ci.reshape(bot_ci.shape[0],-1)
    means = means.reshape(means.shape[0],-1)
    top_ci = top_ci.reshape(top_ci.shape[0],-1)
    my_sd = my_sd.reshape(top_ci.shape[0],-1)
    all_ci_dat = (np.concatenate((labs_target,bot_ci, means, top_ci,my_sd),axis=1))

    if write_output == True:

        with open(str(job_parameters["job_name"])+ set_name + "_confidence_int.csv", "w") as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(all_ci_dat) + 1):
                if i == 0:
                    csvwriter.writerow(
                        [
                            "ids",
                            "target",
                            "bottom_confidence_interval",
                            "mean_output",
                            "top_confidence_interval",
                            "Standard_Deviation"
                        ]
                    )
                elif i > 0:
                    csvwriter.writerow(all_ci_dat[i - 1, :])
                    
def count_ids(dataset, train_sets):
    
    if dataset is not None:

        tot_ids = dataset.indices
        my_ids = []

        for i in range(0, len(train_sets)):
            my_ids = my_ids + train_sets[i].indices

        frame_dict = {'ids': np.unique(tot_ids).tolist(), 'count': np.repeat(0,len(np.unique(tot_ids).tolist()))}
        freq = pd.DataFrame(frame_dict)

        for item in my_ids: 
            if (item in freq['ids'].tolist()): 
                freq['count'][freq['ids'] == item] += 1
            else: 
                print('oopsie')


        with open("bootstrap_freq_training_ids.csv", "w") as f:



            csvwriter = csv.writer(f)
            for i in range(0, len(all_ci_dat) + 1):
                if i == 0:
                    csvwriter.writerow(
                        [
                            "ids",
                            'count'
                        ]
                    )
                elif i > 0:
                    csvwriter.writerow(frq[i - 1, :])








#Train a two models one for the data and one for the model errors and find an alpha non-comformity factors to model errors
#using a regular train test split

def train_ic_errors(
    rank,
    world_size,
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
    processing_args=None
):
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size

    ##Get dataset
    dataset = process.get_dataset(data_path, training_parameters["target_index"], False)



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
    ) = loader_setup(
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
    model = model_setup(
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
    model = trainer(
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
        train_error, train_out = evaluate(
            train_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Train Error: {:.5f}".format(train_error))

        ##Get val error
        if val_loader != None:
            val_error, val_out = evaluate(
                val_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Val Error: {:.5f}".format(val_error))

        ##Get test error
        if test_loader != None:
            test_error, test_out = evaluate(
                test_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Test Error: {:.5f}".format(test_error))

        ##Save model
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
        target_errors[['index','error']].to_csv(os.path.join(os.getcwd(),data_path,'error_targets.csv'), index = False, header=False)


        new_data = process.get_dataset_error(data_path ,training_parameters["target_index"], False, processing_args)
        error_train_subset =  torch.utils.data.Subset(new_data, train_dataset.indices)
        error_val_subset = torch.utils.data.Subset(new_data, val_dataset.indices)
        error_test_subset = torch.utils.data.Subset(new_data, test_dataset.indices)

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

        model_errors = model_setup(
                rank,
                model_parameters["model"],
                model_parameters,
                new_data,
                job_parameters["load_model"],
                job_parameters["model_path"],
                model_parameters.get("print_model", True),
            ) 

        ##Set-up optimizer & scheduler
        optimizer = getattr(torch.optim, model_parameters["optimizer"])(
            model_errors.parameters(),
            lr=model_parameters["lr"],
            **model_parameters["optimizer_args"]
        )
        scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
            optimizer, **model_parameters["scheduler_args"]
        )

        ##Start training
        model_errors = trainer(
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

        train_error_e, train_out_e = evaluate(
            train_loader_e, model_errors, training_parameters["loss"], rank, out=True)

        val_error_e, val_out_e = evaluate(
            val_loader_e, model_errors, training_parameters["loss"], rank, out=True)

        test_error_e, test_out_e = evaluate(
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
        target_val_e_2['alpha'] = target_val_e_2['target_error']/target_val_e_2['predicted_error']

        target_val_e_2 = target_val_e_2.sort_values(['alpha'], axis=0, ascending=True)
        alpha = np.percentile(target_val_e_2['alpha'], 95)

        target_train_e['predicted_error'] = target_train_e['predicted_error'].apply(float)
        target_train_e['lower_error_confidence_level'] = target_train_e['predicted_error'] - alpha
        target_train_e['upper_error_confidence_level'] = target_train_e['predicted_error'] + alpha

        target_val_e['predicted_error'] = target_val_e['predicted_error'].apply(float)
        target_val_e['lower_error_confidence_level'] = target_val_e['predicted_error'] - alpha
        target_val_e['upper_error_confidence_level'] = target_val_e['predicted_error'] + alpha

        target_test_e['predicted_error'] = target_test_e['predicted_error'].apply(float)
        target_test_e['lower_error_confidence_level'] = target_test_e['predicted_error'] - alpha
        target_test_e['upper_error_confidence_level'] = target_test_e['predicted_error'] + alpha


        target_train_e.to_csv(os.path.join(os.getcwd(),data_path,'error_prediction_conf_train.csv'), index = False, header=False)
        target_val_e.to_csv(os.path.join(os.getcwd(),data_path,'error_prediction_conf_val.csv'), index = False, header=False)
        target_test_e.to_csv(os.path.join(os.getcwd(),data_path,'error_prediction_conf_test.csv'), index = False, header=False)

                ##Write outputs
        if job_parameters["write_output"] == "True":

            write_results(
                train_out, str(job_parameters["job_name"]) + "_train_outputs.csv"
            )
            if val_loader != None:
                write_results(
                    val_out, str(job_parameters["job_name"]) + "_val_outputs.csv"
                )
            if test_loader != None:
                write_results(
                    test_out, str(job_parameters["job_name"]) + "_test_outputs.csv"
                )

        if rank not in ("cpu", "cuda"):
            dist.destroy_process_group()

        ##Write out model performance to file
        error_values = np.array((train_error.cpu(), val_error.cpu(), test_error.cpu()))
        if job_parameters.get("write_error") == "True":
            np.savetxt(
                job_parameters["job_name"] + "_errorvalues.csv",
                error_values[np.newaxis, ...],
                delimiter=",",
            )

        return error_values


###Predict using a saved movel
def predict(dataset, loss, job_parameters=None):

    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##Loads predict dataset in one go, care needed for large datasets)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    ##Load saved model
    assert os.path.exists(job_parameters["model_path"]), "Saved model not found"
    if str(rank) == "cpu":
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cpu")
        )
    else:
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cuda")
        )
    model = saved["full_model"]
    model = model.to(rank)
    model_summary(model)

    ##Get predictions
    time_start = time.time()
    test_error, test_out = evaluate(loader, model, loss, rank, out=True)
    elapsed_time = time.time() - time_start

    print("Evaluation time (s): {:.5f}".format(elapsed_time))

    ##Write output
    if job_parameters["write_output"] == "True":
        write_results(
            test_out, str(job_parameters["job_name"]) + "_predicted_outputs.csv"
        )

    return test_error
