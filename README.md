# Convolutional Recurrent Neural Network + CTCLoss 

I think i have fixed the ctcloss nan problem!

Now!

Please pull the latest code from master.

Please update the pytorch to  `>= v1.2.0`

Enjoy it!

> PS: Once there is ctclossnan, please
> 1. Change the `batchSize` to smaller (eg: 8, 16, 32)
> 2. Change the `lr` to smaller (eg: 0.00001, 0.0001)

## Dependence

- CentOS7
- Python3.6.5
- torch==1.2.0
- torchvision==0.4.0
- Tesla P40 - Nvidia

## Run demo
- Run demo

  ```sh
  python demo.py -m path/to/model -i data/demo.jpg
  ```
  

## Feature

- Variable length

  It support variable length.



- Change CTCLoss from [warp-ctc](https://github.com/SeanNaren/warp-ctc) to [torch.nn.CTCLoss](https://pytorch.org/docs/stable/nn.html#ctcloss)

  As we know, warp-ctc need to compile and it seems that it only support PyTorch 0.4. But PyTorch support CTCLoss itself, so i change the loss function to `torch.nn.CTCLoss` .

  

- Solved PyTorch CTCLoss become `nan` after several epoch

  Just don't know why, but when i train the net, the loss always become `nan` after several epoch.

  I add a param `dealwith_lossnan` to `params.py` . If set it to `True` , the net will autocheck and replace all `nan/inf` in gradients to zero.



- DataParallel

  I add a param `multi_gpu` to `params.py` . If you want to use multi gpu to train your net, please set it to `True` and set the param `ngpu` to a proper number.



## Train your data

### Prepare data

The data-loader expects the IAM dataset (or any other dataset that is compatible with it) in the data/ directory. Follow these instructions to get the dataset:

  1. Register for free at this [website](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).
  
  2. Download words/words.tgz.
  
  3. Download ascii/words.txt.
  
  4. Put words.txt into the data/ directory.
  
  5. Create the directory data/words/.
  
  6. Put the content (directories a01, a02, ...) of words.tgz into data/words/.
  
  7. Go to data/ and run python checkDirs.py for a rough check if everything is ok.


### Change parameters and alphabets

Parameters and alphabets can't always be the same in different situation. 

- Change parameters

  Your can see the `params.py` in detail.

- Change alphabets

  Please put all the alphabets appeared in your labels to `alphabets.py` , or the program will throw error during training process.



### Train

Run `train.py` by

```sh
python train.py --trainroot path/to/train/dataset --valroot path/to/val/dataset
```



## Reference

[meijieru/crnn.pytorch](<https://github.com/meijieru/crnn.pytorch>)

[Sierkinhane/crnn_chinese_characters_rec](<https://github.com/Sierkinhane/crnn_chinese_characters_rec>)
