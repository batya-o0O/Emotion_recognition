import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch_multimodal
import time

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

This script trains a model on a video dataset using PyTorch.
"""




if __name__ == '__main__':
    opt = parse_opts()  # Parse command line options
    n_folds = 1  # Number of cross-validation folds
    test_accuracies = []  # List to store test accuracies
    
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Set device to CUDA if available, else CPU

    pretrained = opt.pretrain_path != 'None'  # Check if pre-trained model is specified
    
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)  # Create result directory if it doesn't exist
        
    opt.arch = '{}'.format(opt.model)  # Set architecture name
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])  # Set store name
    
    for fold in range(n_folds):
        print(opt)
        with open(os.path.join(opt.result_path, 'opts'+str(time.time())+str(fold)+'.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)  # Save options to a JSON file
            
        torch.manual_seed(opt.manual_seed)  # Set random seed
        model, parameters = generate_model(opt)  # Generate the model
        
        criterion = nn.CrossEntropyLoss()  # Define the loss function
        criterion = criterion.to(opt.device)  # Move the loss function to the device
        
        if not opt.no_train:
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])  # Define the video transformations
            
            training_data = get_training_set(opt, spatial_transform=video_transform)  # Get the training dataset
            
            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True)  # Create the training data loader
            
            train_logger = Logger(
                os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
                ['epoch', 'loss', 'prec1', 'prec5', 'lr'])  # Logger for training statistics
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])  # Logger for training batch statistics

            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)  # Define the optimizer
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)  # Learning rate scheduler
        
        if not opt.no_val:
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])  # Define the video transformations     
        
            validation_data = get_validation_set(opt, spatial_transform=video_transform)  # Get the validation dataset
            
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)  # Create the validation data loader
        
            val_logger = Logger(
                    os.path.join(opt.result_path, 'val'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])  # Logger for validation statistics
            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])  # Logger for test statistics

        best_prec1 = 0  # Best validation accuracy
        best_loss = 1e10  # Best validation loss
        
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])  # Load checkpoint if specified

        for i in range(opt.begin_epoch, opt.n_epochs + 1):
            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)  # Adjust learning rate
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger)  # Train for one epoch
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                    }
                save_checkpoint(state, False, opt, fold)  # Save checkpoint
            
            if not opt.no_val:
                validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger)  # Validate the model
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
                save_checkpoint(state, is_best, opt, fold)  # Save checkpoint if it's the best so far

        if opt.test:
            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])  # Logger for test statistics

            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])  # Define the video transformations
                
            test_data = get_test_set(opt, spatial_transform=video_transform)  # Get the test dataset
        
            # Load best model
            best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+str(fold)+'.pth')
            model.load_state_dict(best_state['state_dict'])
        
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)  # Create the test data loader
            
            test_loss, test_prec1 = val_epoch(10000, test_loader, model, criterion, opt,
                                            test_logger)  # Test the model
            
            with open(os.path.join(opt.result_path, 'test_set_bestval'+str(fold)+'.txt'), 'a') as f:
                    f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1)  # Append test accuracy to the list
            
    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
        f.write('Prec1: ' + str(np.mean(np.array(test_accuracies))) +'+'+str(np.std(np.array(test_accuracies))) + '\n')  # Write mean and standard deviation of test accuracies to a file
