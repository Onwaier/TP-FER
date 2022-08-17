# TP-FER: An Effective Three-phase Noise-tolerant Recognizer for Facial Expression Recognition
Official implementation about TP-FER

## Train
We train TTL with Torch 1.9.0 and torchvision 0.10.0

**Dataset**
Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), and make sure that it has the same structure as bellow:
```plain
- /your_path/
            - list_patition_label.txt
            - aligned
                train_00001_aligned.jpg
                ...
                test_0001_aligned.jpg
                ...
```


Download [FER2013](https://drive.google.com/file/d/1nJuuij6d80oTs6Tfjez7KeZClM2Y1hvZ/view?usp=sharing), and make sure that it has the same structure as bellow:
```plain
- /your_path/
            - fer2013.h5
```

Download [AffectNet](http://mohammadmahoor.com/affectnet/), and make sure that it has the same structure as bellow:
```plain
- /your_path/
    - Manually_Annotated_file_lists
        - training.csv
        - validation.csv
    - Manually_Annotated_compressed/Manually_Annotated_Images_Align
        - 1
            - 7ffb654b8d3827c453b4a7ffcebd4e4475e33c9097a047d45d38244a.jpg
            - fd9a175d28d67f44f0d277c23509894e4cad1a17d62dff10094b24e2.JPG
            ...
        - 2
        - 3
        ...
    - Manually_Annotated_compressed/Manually_Annotated_Images
        - 1
            - 7ffb654b8d3827c453b4a7ffcebd4e4475e33c9097a047d45d38244a.jpg
            - fd9a175d28d67f44f0d277c23509894e4cad1a17d62dff10094b24e2.JPG
            ...
        - 2
        - 3
        ...
```

**Pretrained model**
Download the pretrained ResNet18 from [this](https://drive.google.com/file/d/12V8HugDD59VOCUIP8YTO-1ITrxgtNNbE/view?usp=sharing).

**Train the model with RAF-DB**

note: <font color='red'>Before running this script, you need to prepare the dataset and pre-trained file, and modify the corresponding path in the code.</font> 
```plain
python train.py --setup_seed 7777 --warm_epoch 5 --relabel_epoch 8 --alpha1 0.60 --alpha2 0.80 --dataset RAF --epochs 100 --batch_size 64 --lr 0.01 --gpu 1 --num_classes 7 --transform_type 0
```

`./log/[02-19]-[16-28]-[58]-log.txt` file records the details of my training on the RAF-DB dataset.

**Train the model with FER2013**

note: <font color='red'>Before running this script, you need to prepare the dataset and pre-trained file, and modify the corresponding path in the code.</font> 

```plain
python train.py --setup_seed 7777 --warm_epoch 95 --relabel_epoch 100 --alpha1 0.60 --alpha2 0.80 --dataset FERPlus --epochs 200 --batch_size 64 --lr 0.01 --gpu 3 --num_classes 7 --transform_type 1 --optimizer sgd --scheduler step --step_size 5
```

`./log/[03-06]-[12-32]-[45]-log.txt` file records the details of my training on the FER2013 dataset.

**Train the model with AffectNet**

note: <font color='red'>Before running this script, you need to prepare the dataset and pre-trained file, and modify the corresponding path in the code.</font> 

```plain
python train.py --setup_seed 7777 --warm_epoch 1000 --relabel_epoch 2 --alpha1 0.60 --alpha2 0.80 --dataset Affect --epochs 10 --batch_size 64 --lr 0.0005 --gpu 0 --num_classes 8 --with_align --transform_type 1
```

`./log/[02-20]-[17-59]-[57]-log.txt` file records the details of my training on the AffectNet dataset.

## Test
Download the checkpoint file about the RAF-DB dataset from [this](https://drive.google.com/file/d/1n0evGFPBRWZ-KOxHVomwG3k5xqNU7AT2/view?usp=sharing) and put it into the `checkpoints` directory.

```plain
cd scripts
python get_acc.py  --dataset RAF --checkpoint_path \[02-19\]-\[16-28\]-\[58\]-model_best.pth
```
