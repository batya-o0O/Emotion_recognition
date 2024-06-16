import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy

def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    # Function for performing validation during training with multimodal inputs
    # modality: specifies which modality to evaluate ('both', 'audio', 'video')
    # dist: specifies the distortion to apply to the non-evaluated modality ('noise', 'addnoise', 'zeros')

    print('validation at epoch {}'.format(epoch))
    assert modality in ['both', 'audio', 'video']    
    model.eval()

    batch_time = AverageMeter()  # Meter for tracking batch processing time
    data_time = AverageMeter()  # Meter for tracking data loading time
    losses = AverageMeter()  # Meter for tracking losses
    top1 = AverageMeter()  # Meter for tracking top-1 accuracy
    top5 = AverageMeter()  # Meter for tracking top-5 accuracy

    end_time = time.time()  # Start time for measuring batch processing time
    for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)  # Update data loading time

        if modality == 'audio':
            print('Skipping video modality')
            if dist == 'noise':
                print('Evaluating with full noise')
                inputs_visual = torch.randn(inputs_visual.size())  # Apply full noise to video inputs
            elif dist == 'addnoise':
                print('Evaluating with noise')
                inputs_visual = inputs_visual + (torch.mean(inputs_visual) + torch.std(inputs_visual) * torch.randn(inputs_visual.size()))  # Add noise to video inputs
            elif dist == 'zeros':
                inputs_visual = torch.zeros(inputs_visual.size())  # Set video inputs to zeros
            else:
                print('UNKNOWN DIST!')
        elif modality == 'video':
            print('Skipping audio modality')
            if dist == 'noise':
                print('Evaluating with noise')
                inputs_audio = torch.randn(inputs_audio.size())  # Apply noise to audio inputs
            elif dist == 'addnoise':
                print('Evaluating with added noise')
                inputs_audio = inputs_audio + (torch.mean(inputs_audio) + torch.std(inputs_audio) * torch.randn(inputs_audio.size()))  # Add noise to audio inputs
            elif dist == 'zeros':
                inputs_audio = torch.zeros(inputs_audio.size())  # Set audio inputs to zeros

        inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)  # Permute dimensions of video inputs
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0] * inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])  # Reshape video inputs

        targets = targets.to(opt.device)  # Move targets to the specified device
        with torch.no_grad():
            inputs_visual = Variable(inputs_visual)  # Wrap video inputs in a Variable (deprecated in newer versions of PyTorch)
            inputs_audio = Variable(inputs_audio)  # Wrap audio inputs in a Variable (deprecated in newer versions of PyTorch)
            targets = Variable(targets)  # Wrap targets in a Variable (deprecated in newer versions of PyTorch)

        outputs = model(inputs_audio, inputs_visual)  # Forward pass through the model
        loss = criterion(outputs, targets)  # Compute the loss
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))  # Compute top-1 and top-5 accuracy
        top1.update(prec1, inputs_audio.size(0))  # Update top-1 accuracy meter
        top5.update(prec5, inputs_audio.size(0))  # Update top-5 accuracy meter

        losses.update(loss.data, inputs_audio.size(0))  # Update loss meter

        batch_time.update(time.time() - end_time)  # Update batch processing time
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  top5=top5))

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})

    return losses.avg.item(), top1.avg.item()