#!/usr/bin/python
# encoding: utf-8

import torch
import params
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import collections
from torch.utils.data.sampler import SubsetRandomSampler


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self):
        # self._ignore_case = ignore_case
        # if self._ignore_case:
        #     alphabet = alphabet.lower()
        # self.alphabet = alphabet + '-'  # for `-1` index

        # self.dict = {}
        # for i, char in enumerate(alphabet):
        #     # NOTE: 0 is reserved for 'blank' required by wrap_ctc
        #     self.dict[char] = i + 1
    
        self.eng_alphabets = params.alphabet
        self.pad_char = '-PAD-'

        self.eng_alpha2index = {self.pad_char: 0}
        for index, alpha in enumerate(self.eng_alphabets):
            self.eng_alpha2index[alpha] = index+1

    def word_rep(self, word, letter2index, max_out_chars, device = 'cpu'):
        rep = torch.zeros(max_out_chars).to(device)
        if max_out_chars < len(word) + 1:
            for i in range(max_out_chars):
                pos = letter2index[word[i]]
                rep[i] = pos
            return(rep ,max_out_chars)
        for letter_index, letter in enumerate(word):
            pos = letter2index[letter]
            rep[letter_index] = pos
        pad_pos = letter2index[self.pad_char]
        rep[letter_index+1] = pad_pos
        return(rep, len(word))

    def words_rep(self, labels_str, max_out_chars = 20, batch_size = params.batchSize, device = 'cpu'):
        words_rep = []
        output_cat = None
        output_2 = None
        lengths_tensor = None
        lengths = []
        for i, label in enumerate(labels_str):
            rep, lnt = self.word_rep(label, self.eng_alpha2index, max_out_chars, device)
            words_rep.append(rep)
            if lengths_tensor is None:
                lengths_tensor = torch.empty(len(labels_str), dtype = torch.long)
            if output_cat is None:
                output_cat_size = list(rep.size())
                output_cat_size.insert(0, len(labels_str))
                output_cat = torch.empty(*output_cat_size, dtype=rep.dtype, device=rep.device)
            if output_2 is None:
                output_2 = rep[:lnt]
            else:
                output_2 = torch.cat([output_2, rep[:lnt]], dim = 0)

            output_cat[i, :] = rep
            lengths_tensor[i] = lnt
            lengths.append(lnt)
        return(output_cat, lengths_tensor)

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.
        """

        length = []
        result = []
        for item in text:            
            item = item.decode('utf-8','strict')
            length.append(len(item))
            r = []
            for char in item:
                index = self.dict[char]
                # result.append(index)
                r.append(index)
            result.append(r)
        
        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)
        
        result_temp = []
        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)

        text = result_temp
        return (torch.LongTensor(text), torch.LongTensor(length))


    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def dataloader(dataset, batch_size, validation_split, shuffle_dataset):
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return(train_loader, val_loader)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img
