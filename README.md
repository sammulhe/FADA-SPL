
## Datasets
- [SSC](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)
- [HAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [HHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO)
- [WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B)
- [MFD](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN)

Data directory structure
```
.
└── data
    └── HAR
        ├── test_0.pt
        ├── test_1.pt
        ├── test_2.pt
        ├── test_3.pt
        ├── test_4.pt
        ├── train_0.pt
        ├── train_1.pt
        ├── train_2.pt
        ├── train_3.pt
        └── train_4.pt
    
    └── HHAR
      ......
    └── WISDM
      ......
```

## How to Run
For each dataset, we select **8** source-target domain pairs.
Each experiment is repeated **5** times with different random seeds.


To train a model on UCIHAR dataset:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
 --experiment_description FDA \
 --run_description UCIHAR \
 --da_method FDA \
 --dataset UCIHAR \
 --num_runs 5 \
 --lr 0.001 \
 --cls_trade_off 1 \
 --domain_trade_off 1 \
 --entropy_trade_off 0.1 \
```
