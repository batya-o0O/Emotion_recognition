from torch import nn
from models import multimodalcnn

'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''



def generate_model(opt):
    # Check if the specified model is supported
    assert opt.model in ['multimodalcnn']

    if opt.model == 'multimodalcnn':
        # Create an instance of the MultiModalCNN model
        model = multimodalcnn.MultiModalCNN(opt.n_classes, fusion=opt.fusion, seq_length=opt.sample_duration, pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)

    if opt.device != 'cpu':
        # Move the model to the specified device
        model = model.to(opt.device)
        # Use DataParallel to parallelize the model across multiple GPUs
        model = nn.DataParallel(model, device_ids=None)
        # Calculate the total number of trainable parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

    # Return the model and its parameters
    return model, model.parameters()
