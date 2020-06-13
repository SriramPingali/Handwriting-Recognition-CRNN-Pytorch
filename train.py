from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import utils
import dataset

import models.crnn as net
import params

# parser = argparse.ArgumentParser()
# parser.add_argument('-train', '--trainroot', required=True, help='path to train dataset')
# parser.add_argument('-val', '--valroot', required=True, help='path to val dataset')
# args = parser.parse_args()

# if not os.path.exists(params.expr_dir):
#     os.makedirs(params.expr_dir)

# ensure everytime the random is the same
random.seed(params.manualSeed)
np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
cudnn.benchmark = True

if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")

# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""
class dataset(Dataset):

    def __init__(self, image_root, label_root, img_x, img_y):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """
        self.images_path = image_root
        self.labels_path = label_root
        self.data_len = 0
        self.images = []
        self.labels = open(self.labels_path, "r").readlines()
        self.transform = transforms.Compose([
            transforms.Resize((img_x, img_y)),  
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        for root, dirs, files in os.walk(self.images_path):
            for file in files:
                if file.endswith('.png'):
                    self.data_len += 1
                    temp = file.split("-")
                    self.images.append(self.images_path + temp[0] + '/' + temp[0] + "-" + temp[1] + "/" + file)

    def __len__(self):
        """return number of points in our dataset"""
        return(self.data_len)

    def __getitem__(self, idx):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """
        img = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img)
        img = img.convert('RGB')
        img = self.transform(img)
        return(img, label[:-1])
    
def loader_param():
    img_x = 32
    img_y = 128
    batch_size = 4
    return(img_x, img_y, batch_size)

img_x, img_y, batch_size = loader_param()
train_set = dataset(image_root="../IAM Dataset/words/", label_root = "../IAM Dataset/ascii/labels.txt", img_x = img_x, img_y = img_y)
# train_set.__getitem__(345)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.batchSize, shuffle=True, num_workers=0)
val_loader = train_loader

eng_alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"!#&\'()*+,-./0123456789:;?'
pad_char = '-PAD-'

eng_alpha2index = {pad_char: 0}
for index, alpha in enumerate(eng_alphabets):
    eng_alpha2index[alpha] = index+1

def word_rep(word, letter2index, max_out_chars, device = 'cpu'):
    rep = torch.zeros(max_out_chars).to(device)
    if max_out_chars < len(word) + 1:
        for i in range(max_out_chars):
            pos = letter2index[word[i]]
            rep[i] = pos
        return(rep ,max_out_chars)
    for letter_index, letter in enumerate(word):
        pos = letter2index[letter]
        rep[letter_index] = pos
    pad_pos = letter2index[pad_char]
    rep[letter_index+1] = pad_pos
    return(rep, len(word))

def words_rep(labels_str, max_out_chars = 20, batch_size = params.batchSize):
    words_rep = []
    output_cat = None
    output_2 = None
    lengths_tensor = None
    lengths = []
    for i, label in enumerate(labels_str):
        rep, lnt = word_rep(label, eng_alpha2index, max_out_chars, device)
        words_rep.append(rep)
#         print(rep[0:lnt])
        if lengths_tensor is None:
            lengths_tensor = torch.empty(len(labels_str), dtype = torch.long)
        if output_cat is None:
            output_cat_size = list(rep.size())
            output_cat_size.insert(0, len(labels_str))
            output_cat = torch.empty(*output_cat_size, dtype=rep.dtype, device=rep.device)
#             print(output_cat.shape)
        if output_2 is None:
            output_2 = rep[:lnt]
        else:
            output_2 = torch.cat([output_2, rep[:lnt]], dim = 0)

        output_cat[i, :] = rep
        lengths_tensor[i] = lnt
        lengths.append(lnt)
#     print(output_2)
    return(output_cat, lengths_tensor)

# def data_loader():
#     # train
#     train_dataset = dataset.lmdbDataset(root=args.trainroot)
#     assert train_dataset
#     if not params.random_sample:
#         sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
#     else:
#         sampler = None
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batchSize, \
#             shuffle=True, sampler=sampler, num_workers=int(params.workers), \
#             collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    
#     # val
#     val_dataset = dataset.lmdbDataset(root=args.valroot, transform=dataset.resizeNormalize((params.imgW, params.imgH)))
#     assert val_dataset
#     val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    
#     return train_loader, val_loader

# train_loader, val_loader = data_loader()

# -----------------------------------------------
"""
In this block
    Net init
    Weight init
    Load pretrained model
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def net_init():
    nclass = len(params.alphabet) + 1
    crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
    crnn.apply(weights_init)
    if params.pretrained != '':
        print('loading pretrained model from %s' % params.pretrained)
        if params.multi_gpu:
            crnn = torch.nn.DataParallel(crnn)
        crnn.load_state_dict(torch.load(params.pretrained))
    
    return crnn

# crnn = net_init()
crnn = net.CRNN(32, 3, 79, 1024)
print(crnn)

# -----------------------------------------------
"""
In this block
    Init some utils defined in utils.py
"""
# Compute average for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.averager()

# Convert between str and label.
converter = utils.strLabelConverter(params.alphabet)

# -----------------------------------------------
"""
In this block
    criterion define
"""
criterion = CTCLoss()

# -----------------------------------------------
"""
In this block
    Init some tensor
    Put tensor and net on cuda
    NOTE:
        image, text, length is used by both val and train
        becaues train and val will never use it at the same time.
"""
image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

    crnn = crnn.cuda()
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(params.ngpu))

image = Variable(image)
text = Variable(text)
length = Variable(length)

# -----------------------------------------------
"""
In this block
    Setup optimizer
"""
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

# -----------------------------------------------
"""
In this block
    Dealwith lossnan
    NOTE:
        I use different way to dealwith loss nan according to the torch version. 
"""
if params.dealwith_lossnan:
    if torch.__version__ >= '1.1.0':
        """
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.
        Pytorch add this param after v1.1.0 
        """
        criterion = CTCLoss(zero_infinity = True)
    else:
        """
        only when
            torch.__version__ < '1.1.0'
        we use this way to change the inf to zero
        """
        crnn.register_backward_hook(crnn.backward_hook)

# -----------------------------------------------

def val(net, criterion):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager() # The blobal loss_avg is used by train

    max_iter = len(val_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for i in cpu_texts:
            cpu_texts_decode.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, cpu_texts_decode):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * params.batchSize)
    print('Val loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def train(net, criterion, optimizer, train_iter):
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()

    data = train_iter.next()
    cpu_images, cpu_texts = data
    text, length = words_rep(cpu_texts, max_out_chars = 20, batch_size = params.batchSize)
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    # t, l = converter.encode(cpu_texts)
    # utils.loadData(text, t)
    # utils.loadData(length, l)
    
    optimizer.zero_grad()
    preds = crnn(image)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    # print("Label: ", text[0], '\nOutput: ', torch.argmax(preds, 2)[:, 0])
    cost = criterion(preds, text, preds_size, length) / batch_size
    # crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == "__main__":
    for epoch in range(params.nepoch):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            cost = train(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1

            if i % params.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, params.nepoch, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            # if i % params.valInterval == 0:
            #     val(crnn, criterion)

            # do checkpointing
            if i % params.saveInterval == 0:
                torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(params.expr_dir, epoch, i))
