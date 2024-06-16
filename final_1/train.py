import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy

'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    # Print the current epoch
    print('train at epoch {}'.format(epoch))
    
    # Set the model to training mode
    model.train()

    # Initialize average meters to keep track of metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
        
    # Record the start time of the epoch
    end_time = time.time()
    
    # Iterate over the data loader
    for i, (audio_inputs, visual_inputs, targets) in enumerate(data_loader):
        # Measure the time taken to load the data
        data_time.update(time.time() - end_time)

        # Move the targets to the specified device (e.g., GPU)
        targets = targets.to(opt.device)
            
        if opt.mask is not None:
            with torch.no_grad():
                if opt.mask == 'noise':
                    # Add noise to the audio and visual inputs
                    audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)                   
                    visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0) 
                    targets = torch.cat((targets, targets, targets), dim=0)                    
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
                    
                elif opt.mask == 'softhard':
                    # Generate random coefficients for audio and visual inputs
                    coefficients = torch.randint(low=0, high=100,size=(audio_inputs.size(0),1,1))/100
                    vision_coefficients = 1 - coefficients
                    coefficients = coefficients.repeat(1,audio_inputs.size(1),audio_inputs.size(2))
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,visual_inputs.size(1), visual_inputs.size(2), visual_inputs.size(3), visual_inputs.size(4))

                    # Apply soft-hard masking to the audio and visual inputs
                    audio_inputs = torch.cat((audio_inputs, audio_inputs*coefficients, torch.zeros(audio_inputs.size()), audio_inputs), dim=0) 
                    visual_inputs = torch.cat((visual_inputs, visual_inputs*vision_coefficients, visual_inputs, torch.zeros(visual_inputs.size())), dim=0)   
                    
                    targets = torch.cat((targets, targets, targets, targets), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
   
        # Reshape the visual inputs for efficient computation
        visual_inputs = visual_inputs.permute(0,2,1,3,4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0]*visual_inputs.shape[1], visual_inputs.shape[2], visual_inputs.shape[3], visual_inputs.shape[4])
        
        # Wrap the inputs and targets in Variables
        audio_inputs = Variable(audio_inputs)
        visual_inputs = Variable(visual_inputs)
        targets = Variable(targets)
        
        # Forward pass through the model
        outputs = model(audio_inputs, visual_inputs)
        
        # Compute the loss
        loss = criterion(outputs, targets)

        # Update the average meters
        losses.update(loss.data, audio_inputs.size(0))
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, audio_inputs.size(0))
        top5.update(prec5, audio_inputs.size(0))

        # Zero the gradients, perform backpropagation, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the batch time and record the end time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # Log the metrics for the current batch
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Print the metrics every 10 batches
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    # Log the metrics for the current epoch
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

 
def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    # Print the current epoch
    print('train at epoch {}'.format(epoch))
    
    # Check if the model is multimodal
    if opt.model == 'multimodalcnn':
        # Call the train_epoch_multimodal function
        train_epoch_multimodal(epoch,  data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger)
        return
