from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from jersey_number_dataset import JerseyNumberLegibilityDataset, UnlabelledJerseyNumberLegibilityDataset, TrackletLegibilityDataset
from networks import LegibilityClassifier, LegibilitySimpleClassifier, LegibilityClassifier34, LegibilityClassifier50, LegibilityClassifierTransformer

import time
import datetime
import copy
import argparse
import os
import configuration as cfg
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

from sam.sam import SAM

standard_batch_size = 64
standard_worker_num = 4

log_file="C:\\Users\\Riley\\Documents\\UBC\\GitHub\\COSC519-Jersey-Number-Recognition\\legibility_log.txt"

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                #print(f"input and label sizes:{len(inputs), len(labels)}")
                labels = labels.reshape(-1, 1)
                labels = labels.type(torch.FloatTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(f"output size is {len(outputs)}")
                    preds = outputs.round()
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train_model_with_sam(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        epoch_val_acc = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            pbar = tqdm(dataloaders[phase], desc=f"Epoch {epoch} {phase.capitalize()}")

            for inputs, labels, _ in pbar:
                labels = labels.reshape(-1, 1).type(torch.FloatTensor).to(device)
                inputs = inputs.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        # First forward pass
                        outputs = model(inputs) 
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.first_step(zero_grad=True)

                        # Second forward-backward pass
                        model.train()
                        # Calculate output again for the second pass loss
                        criterion(model(inputs), labels).backward()
                        optimizer.second_step(zero_grad=True)
                    else: # Validation forward pass
                         outputs = model(inputs)
                         loss = criterion(outputs, labels)

                # Statistics calculation
                with torch.no_grad():
                    preds = outputs.round()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                pbar.set_postfix({
                    'Loss': f'{running_loss / ((pbar.n + 1) * inputs.size(0)):.4f}',
                    'Acc': f'{running_corrects.double() / ((pbar.n + 1) * inputs.size(0)):.4f}'
                 })

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                epoch_val_acc = epoch_acc
                if epoch_acc > best_acc:
                    print(f"Validation accuracy improved from {best_acc:.4f} to {epoch_acc:.4f}. Saving model weights.")
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # Scheduler step
        if scheduler is not None:
             if isinstance(scheduler, ReduceLROnPlateau):
                 scheduler.step(epoch_val_acc)
             else:
                 scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

def run_full_validation(model, dataloader):
    results = []
    tracks = []
    gt = []
    # # load weights
    # state_dict = torch.load(load_model_path, map_location=device)
    # current_model_dict = model.state_dict()
    # new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), state_dict.values())}
    # model.load_state_dict(new_state_dict, strict=False)
    # model_ft = model.to(device)
    for inputs, track, label in dataloader:
        # print(f"input and label sizes:{len(inputs), len(labels)}")
        inputs = inputs.to(device)

        # zero the parameter gradients
        torch.set_grad_enabled(False)
        outputs = model(inputs)

        outputs = outputs.float()

        preds = outputs.cpu().detach().numpy()
        flattened_preds = preds.flatten().tolist()
        results += flattened_preds
        tracks += track
        gt += label

    # evaluate tracklet-level accuracy
    unique_tracks = np.unique(np.array(tracks))
    result_dict = {key:[] for key in unique_tracks}
    track_gt = {key:0 for key in unique_tracks}
    for i, result in enumerate(results):
        result_dict[tracks[i]].append(round(result))
        track_gt[tracks[i]] = gt[i]
    correct = 0
    total = 0
    for track in result_dict.keys():
        if not track.isnumeric():
            continue
        legible = list(np.nonzero(result_dict[track]))[0]
        if len(legible) == 0 and track_gt[track] == 0:
            correct += 1
        elif len(legible) > 0 and track_gt[track] == 1:
            correct += 1
        total += 1

    # Calculate final accuracy
    if total == 0:
        accuracy = 0 # Avoid division by zero if no valid tracks were processed
        print("No tracks evaluated")
    else:
        accuracy = correct / total

    # Attempt to write to file
    try:
        # Creates the file if it doesn't exist
        with open(log_file, 'a') as f:
            # Write the calculated accuracy (6 decimal places)
            f.write(f"{accuracy:.6f}\n")
            print(f"Accuracy {accuracy:.6f} appended to {log_file}")
    # Handle file writing errors
    except IOError as e:
        print(f"Error: Could not write to log file {log_file}. Reason: {e}")

    return accuracy

def train_model_with_sam_and_full_val(model, criterion, optimizer, scheduler, num_epochs=25, early_stopping_threshold=1.0):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    stop_training = False

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        epoch_full_val_acc = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            if phase == 'train':
                running_loss = 0.0
                running_corrects = 0
                pbar = tqdm(dataloaders[phase], desc=f"Epoch {epoch} Train")

                for inputs, labels, _ in pbar:
                    labels = labels.reshape(-1, 1).type(torch.FloatTensor).to(device)
                    inputs = inputs.to(device)

                    # Ensures gradients are only tracked during training
                    with torch.set_grad_enabled(True): 
                        # First forward pass
                        outputs = model(inputs) 
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.first_step(zero_grad=True)

                        # Second forward-backward pass
                        model.train()
                        criterion(model(inputs), labels).backward()
                        optimizer.second_step(zero_grad=True)

                    # Statistics calculation
                    with torch.no_grad():
                        preds = outputs.round()
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    pbar.set_postfix({
                        'Loss': f'{running_loss / ((pbar.n + 1) * inputs.size(0)):.4f}',
                        'Acc': f'{running_corrects.double() / ((pbar.n + 1) * inputs.size(0)):.4f}'
                    })

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc_train = running_corrects.double() / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc_train:.4f}')

            # phase == "val"
            else: 
                model.eval()
                print("Running full validation...")
                val_acc = run_full_validation(model, dataloaders['val'])
                epoch_full_val_acc = val_acc
                print(f'{phase} Full Tracklet Acc: {val_acc:.4f}')

                if val_acc > best_acc:
                    print(f"Validation accuracy improved from {best_acc:.4f} to {val_acc:.4f}. Saving model weights.")
                    best_acc = val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                if val_acc >= early_stopping_threshold:
                    print(f"\nValidation accuracy ({val_acc:.4f}) met or exceeded threshold ({early_stopping_threshold:.4f}).")
                    print("Stopping training early.")
                    stop_training = True


        # Scheduler step
        if scheduler is not None:
             if isinstance(scheduler, ReduceLROnPlateau):
                 scheduler.step(epoch_full_val_acc)
             else:
                 scheduler.step()
        print()

        if stop_training:
            break

    # After Training Loop
    time_elapsed = time.time() - since
    if stop_training:
         print(f'Training stopped early in epoch {epoch} after {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    else:
         print(f'Training completed {num_epochs} epochs in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Full Tracklet Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, subset, result_path=None):
    model.eval()
    running_corrects = 0
    temp_max = 500
    temp_count = 0

    # Store results directly as lists (CPU)
    gt_list = []
    predictions_list = []
    raw_predictions_list = []
    all_img_names = []

    with torch.no_grad():
        for inputs, labels, names in tqdm(dataloaders[subset]):
            # Get actual batch size
            batch_size = inputs.size(0)
            temp_count += batch_size
            inputs = inputs.to(device)
            # Keep labels on CPU if possible, or move copy to GPU for comparison
            labels_cpu = labels.reshape(-1, 1).type(torch.FloatTensor)
            labels_gpu = labels_cpu.to(device)

            outputs = model(inputs)
            preds = outputs.round()

            # Compare on GPU
            running_corrects += torch.sum(preds == labels_gpu)

            # Transfer batch's results to CPU and append to lists
            gt_list.extend(labels_cpu.numpy().flatten().tolist())
            predictions_list.extend(preds.cpu().numpy().flatten().tolist())
            raw_predictions_list.extend(outputs.cpu().numpy().flatten().tolist())
            all_img_names.extend(list(names))

            if subset == 'train' and temp_count >= temp_max:
                break

    # Data is already collected in Python lists on the CPU
    # gt = gt_list
    # predictions = predictions_list
    # raw_predictions = raw_predictions_list
    if subset == 'train':
        epoch_acc = running_corrects.cpu().double() / temp_count if temp_count > 0 else 0.0
    else:
         # Ensure dataset_sizes[subset] is correct size
        epoch_acc = running_corrects.cpu().double() / dataset_sizes[subset]

    return epoch_acc


# run inference on a list of files
def run(image_paths, model_path, threshold=0.5, arch='resnet18'):
    # setup data
    dataset = UnlabelledJerseyNumberLegibilityDataset(image_paths, arch=arch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=standard_batch_size,
                                                  shuffle=False, num_workers=standard_worker_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    #load model
    state_dict = torch.load(model_path, map_location=device)
    if arch == 'resnet18':
        model_ft = LegibilityClassifier()
    elif arch == 'vit':
        model_ft = LegibilityClassifierTransformer()
    elif arch == "resnet50":
        model_ft = LegibilityClassifier50()
    else:
        model_ft = LegibilityClassifier34()

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    model_ft.load_state_dict(state_dict)
    model_ft = model_ft.to(device)
    model_ft.eval()

    # run classifier
    results = []
    for inputs in dataloader:
        # print(f"input and label sizes:{len(inputs), len(labels)}")
        inputs = inputs.to(device)

        # zero the parameter gradients
        torch.set_grad_enabled(False)
        outputs = model_ft(inputs)

        if threshold > 0:
            outputs = (outputs>threshold).float()
        else:
            outputs = outputs.float()
        preds = outputs.cpu().detach().numpy()
        flattened_preds = preds.flatten().tolist()
        results += flattened_preds

    return results


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='fine-tune model by loading public IMAGENET-trained weights')
    parser.add_argument('--sam', action='store_true', help='Use Sharpness-Aware Minimization during training')
    parser.add_argument('--finetune', action='store_true', help='load custom fine-tune weights for further training')
    parser.add_argument('--data', help='data root dir')
    parser.add_argument('--trained_model_path', help='trained model to use for testing or to load for finetuning')
    parser.add_argument('--new_trained_model_path', help='path to save newly trained model')
    parser.add_argument('--arch', choices=['resnet18', 'simple', 'resnet50', 'resnet34', 'vit'], default='resnet18', help='what architecture to use')
    parser.add_argument('--full_val_dir', help='to use tracklet instead of images for validation specify val dir')
    parser.add_argument('--early_stop_acc', type=float, default=0.9, help='Validation accuracy threshold for early stopping with --full_val_dir (e.g., 0.95). Default 1.0 (disabled).')
    
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')

    # Arguments for ReduceLROnPlateau
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='Factor by which the learning rate will be reduced by ReduceLROnPlateau.')
    parser.add_argument('--lr_scheduler_patience', type=int, default=2, help='Number of epochs with no improvement after which learning rate will be reduced by ReduceLROnPlateau.')

    args = parser.parse_args()

    annotations_file = '_gt.txt'
    use_full_validation = (not args.full_val_dir is None) and (len(args.full_val_dir) > 0)

    image_dataset_train = JerseyNumberLegibilityDataset(os.path.join(args.data, 'train', 'train' + annotations_file),
                                                        os.path.join(args.data, 'train', 'images'), 'train', isBalanced=True, arch=args.arch)
    if not args.train and not args.finetune:
        image_dataset_test = JerseyNumberLegibilityDataset(os.path.join(args.data, 'test', 'test' + annotations_file),
                                                       os.path.join(args.data, 'test', 'images'), 'test', arch=args.arch)

    dataloader_train = torch.utils.data.DataLoader(image_dataset_train, batch_size=standard_batch_size,
                                                   shuffle=True, num_workers=standard_worker_num)

    if not args.train and not args.finetune:
        dataloader_test = torch.utils.data.DataLoader(image_dataset_test, batch_size=standard_batch_size,
                                                  shuffle=False, num_workers=standard_worker_num)

    # use full validation set during training
    if use_full_validation:
        image_dataset_full_val = TrackletLegibilityDataset(os.path.join(args.full_val_dir, 'val_gt.json'),
                                                          os.path.join(args.full_val_dir, 'images'), arch=args.arch)
        dataloader_full_val = torch.utils.data.DataLoader(image_dataset_full_val, batch_size=standard_batch_size,
                                                     shuffle=False, num_workers=standard_worker_num)
        image_datasets = {'train': image_dataset_train, 'val': image_dataset_full_val}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        dataloaders = {'train': dataloader_train, 'val': dataloader_full_val}

    elif not args.train and not args.finetune:
        image_datasets = {'test': image_dataset_test}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
        dataloaders = {'test': dataloader_test}
    else:
        image_dataset_val = JerseyNumberLegibilityDataset(os.path.join(args.data, 'val', 'val' + annotations_file),
                                                          os.path.join(args.data, 'val', 'images'), 'val', arch=args.arch)
        dataloader_val = torch.utils.data.DataLoader(image_dataset_val, batch_size=standard_batch_size,
                                                     shuffle=True, num_workers=standard_worker_num)
        image_datasets = {'train': image_dataset_train, 'val': image_dataset_val}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        dataloaders = {'train': dataloader_train, 'val': dataloader_val}

    if args.arch == 'resnet18':
        model_ft = LegibilityClassifier()
    elif args.arch == 'simple':
        model_ft = LegibilitySimpleClassifier()
    elif args.arch == 'vit':
        model_ft = LegibilityClassifierTransformer()
    elif args.arch == "resnet50":
        model_ft = LegibilityClassifier50()
    else:
        model_ft = LegibilityClassifier34()

    if args.train or args.finetune:
        print("Begin fine-tune")
        # Print current time
        ct = datetime.datetime.now()
        print(f'current time: ${ct}')
        if args.finetune:
            if args.trained_model_path is None or args.trained_model_path == '':
                load_model_path = cfg.dataset["Hockey"]['legibility_model']
            else:
                load_model_path = args.trained_model_path
            # load weights
            state_dict = torch.load(load_model_path, map_location=device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            current_model_dict = model_ft.state_dict()
            new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), state_dict.values())}
            model_ft.load_state_dict(new_state_dict, strict=False)

        model_ft = model_ft.to(device)
        criterion = nn.BCELoss()
        scheduler_ft = None # Initialize scheduler
        if args.sam:
            # Observe that all parameters are being optimized
            base_optimizer = torch.optim.SGD
            optimizer_ft = SAM(model_ft.parameters(), base_optimizer, lr=0.001, momentum=0.9)
            scheduler_ft = ReduceLROnPlateau(optimizer_ft, mode='max', factor=args.lr_scheduler_factor,
                                             patience=args.lr_scheduler_patience, verbose=True)
            
            if use_full_validation:
                model_ft = train_model_with_sam_and_full_val(model_ft, criterion, optimizer_ft, scheduler_ft,
                                                             num_epochs=args.num_epochs,
                                                             early_stopping_threshold=args.early_stop_acc)
            else:
                model_ft = train_model_with_sam(model_ft, criterion, optimizer_ft, num_epochs=20)
        else:
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
            model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                   num_epochs=15)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_model_path = f"./experiments/legibility_{args.arch}_{timestr}.pth"

        torch.save(model_ft.state_dict(), save_model_path)
        # Print current time
        ct = datetime.datetime.now()
        print(f'current time: ${ct}')
        print(f'Fine-tune complete, model saved at ${save_model_path}')

    else:
        #load weights
        state_dict = torch.load(args.trained_model_path, map_location=device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        model_ft.load_state_dict(state_dict)
        model_ft = model_ft.to(device)

        test_model(model_ft, 'test', result_path=args.raw_result_path)