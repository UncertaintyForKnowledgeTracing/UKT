# [Uncertainty-aware Knowledge Tracing][https://arxiv.org/abs/2501.05415]
<div align=center><img src="https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/main/picture/model.png"/></div>
## Installation

Use the following command to install pyKT:

Create conda envirment.

```
conda create --name=ukt python=3.7.5
conda activate ukt
```

```
pip install -U pykt-toolkit -i  https://pypi.python.org/simple 

```

# Dataset
we use datasets including :

Assist2009(https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)

Algebra2005 (https://pslcdatashop.web.cmu.edu/KDDCup/)

Bridge2006 (https://pslcdatashop.web.cmu.edu/KDDCup/)

NIPS34 (https://eedi.com/projects/neurips-education-challenge)

ASSISTments2015 (https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data)

POJ(https://drive.google.com/drive/folders/1LRljqWfODwTYRMPw6wEJ_mMt1KZ4xBDk)

## Data Preparation

```
cd train
python data_preprocess.py --dataset_name=assist2009
```



## Run Your Model

We provide the Hyper Parameter we use for training, you could run UKT as follows command 

```
CUDA_VISIBLE_DEVICES=0 python wandb_ukt_train.py --fold=0 --emb_type=qid --loss3=0.5 --d_ff=64   --nheads=4 --dropout=0.1 --loss2=0.5 --final_fc_dim2=256 --loss1=0.5 --d_model=256 --num_attn_heads=4 --num_layers=2 --seed=42    --final_fc_dim=512 --n_blocks=4 --start=50 --learning_rate=0.0001  --dataset_name=assist2009 --emb_type='stoc_qid' --atten_type='w2'
```

## Run Baseline Model
You can also use the follows command to run baseline methods.

CUDA_VISIBLE_DEVICES=2 python wandb_akt_train.py --use_wandb=0 --add_uuid=0 --fold=0 --emb_type=qid --d_ff=64   --dropout=0.1   --d_model=256 --num_attn_heads=4  --seed=42   --n_blocks=4  --learning_rate=0.0001  --dataset_name=assist2009 

Hyper Parameter of each baseline can be found at[https://github.com/pykt-team/pykt-toolkit]


## Citation
If you use our code or find UKT useful in your work, please cite our paper as:
```bib
Cheng W, Du H, Li C, et al. Uncertainty-aware Knowledge Tracing[J]. arXiv preprint arXiv:2501.05415, 2025.
```


## Evaluate Your Model

Now, let’s use `wandb_predict.py` to evaluate the model performance on the testing set.

```
python wandb_predict.py --save_dir=saved_model/YourModelPath
```

--save_dir is the save path of your trained model that you can find in your training log

## Baseline_Evaluation
<div align=center><img src="https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/main/result.png"/></div>

