# [Uncertainty-aware Knowledge Tracing]
<div align=center><img src="https://github.com/UncertaintyForKnowledgeTracing/UKT/edit/main/picture/model.pdf"/></div>
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
CUDA_VISIBLE_DEVICES=1 python wandb_ukt_train.py --fold=0 --emb_type=qid --loss3=0.5 --d_ff=256   --nheads=4 --dropout=0.1 --loss2=0.5 --final_fc_dim2=256 --loss1=0.5 --d_model=256 --num_attn_heads=4 --num_layers=2 --seed=42    --final_fc_dim=512 --n_blocks=4 --start=50 --learning_rate=0.0001  --dataset_name=assist2009  --use_CL=1 --cl_weight=0.02 --use_uncertainty_aug=1> output/assist2009/ukt_train 
```
as for other dataset, you can find the Hyper Parameter we use in url(to be added)

## Run Baseline Model
You can also use the follows command to run baseline methods.

CUDA_VISIBLE_DEVICES=2 python wandb_akt_train.py --use_wandb=0 --add_uuid=0 --fold=0 --emb_type=qid --d_ff=64   --dropout=0.1   --d_model=256 --num_attn_heads=4  --seed=42   --n_blocks=4  --learning_rate=0.0001  --dataset_name=assist2009 

Hyper Parameter of each baseline can be found at(to be added)


## Citation
If you use our code or find UKT useful in your work, please cite our paper as:
```bib

```


## Evaluate Your Model

Now, let’s use `wandb_predict.py` to evaluate the model performance on the testing set.

```
python wandb_predict.py --save_dir=saved_model/YourModelPath
```

--save_dir is the save path of your trained model that you can find in your training log

## Baseline_Evaluation
<div align=center><img src="https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/main/picture/result.png"/></div>

