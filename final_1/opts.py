import argparse

# -*- coding: utf-8 -*-
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''



def parse_opts():
    # Create an argument parser
    parser = argparse.ArgumentParser()
    
    # Add command line arguments
    parser.add_argument('--annotation_path', default="C:/Users/zhk27/OneDrive/Рабочий стол/final_1/annotations.txt", type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='results_new', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--dataset', default='RAVDESS', type=str, help='Used dataset. Currently supporting Ravdess')
    parser.add_argument('--n_classes', default=5, type=int, help='Number of classes')
    
    parser.add_argument('--model', default='multimodalcnn', type=str, help='')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads, in the paper 1 or 4')
    
    parser.add_argument('--device', default='cuda', type=str, help='Specify the device to run. Defaults to cuda, fallsback to cpu')
    
    parser.add_argument('--sample_size', default=224, type=int, help='Video dimensions: ravdess = 224 ')
    parser.add_argument('--sample_duration', default=15, type=int, help='Temporal duration of inputs, ravdess = 15')
    
    parser.add_argument('--learning_rate', default=0.04, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 200, 250], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of total epochs to run')
    
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default='C:/Users/zhk27/OneDrive/Рабочий стол/final_1/EfficientFace_Trained_on_AffectNet7.pth.tar', type=str, help='Pretrained model (.pth), efficientface')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument('--test_subset', default='test', type=str, help='Used subset in test (val | test)')
    
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--video_norm_value', default=255, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
 
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--fusion', default='ia', type=str, help='fusion type: lt | it | ia')
    parser.add_argument('--mask', type=str, help='dropout type : softhard | noise | nodropout', default='softhard')
    
    # Parse the command line arguments
    args = parser.parse_args()

    return args





class Opts:
    def __init__(self):
        # Default values
        self.annotation_path = "C:/Users/zhk27/OneDrive/Рабочий стол/final_1/annotations.txt"
        self.result_path = "C:/Users/zhk27/OneDrive/Рабочий стол/final_1/results"
        self.store_name = 'model'
        self.dataset = 'RAVDESS'
        self.n_classes = 5
        self.model = 'multimodalcnn'
        self.num_heads = 1
        self.device = 'cuda'
        self.sample_size = 224
        self.sample_duration = 15
        self.learning_rate = 0.04
        self.momentum = 0.9
        self.lr_steps = [40, 55, 65, 70, 200, 250]
        self.dampening = 0.9
        self.weight_decay = 1e-3
        self.lr_patience = 10
        self.batch_size = 8
        self.n_epochs = 10
        self.begin_epoch = 1
        self.resume_path = ''
        self.pretrain_path = 'C:/Users/zhk27/OneDrive/Рабочий стол/final_1/EfficientFace_Trained_on_AffectNet7.pth.tar'
        self.no_train = False
        self.no_val = False
        self.test = True
        self.test_subset = 'test'
        self.n_threads = 16
        self.video_norm_value = 255
        self.manual_seed = 1
        self.fusion = 'ia'
        self.mask = 'softhard'

# Instantiate the options object
opts = Opts()
