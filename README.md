# UKT


## Installation

Use the following command to install pyKT:

Create conda envirment.

```
conda create --name=ukt python=3.7.5
source activateukt
```

```
pip install -U pykt-toolkit -i  https://pypi.python.org/simple 

```

## Data Preparation

```
cd train
python data_preprocess.py --dataset_name=assist2009
```



## Run

We provide the Hyper Parameter we use for training, please run UKT as follows command

```
CUDA_VISIBLE_DEVICES=1 python wandb_ukt_train.py --fold=0 --emb_type=qid --loss3=0.5 --d_ff=256   --nheads=4 --dropout=0.1 --loss2=0.5 --final_fc_dim2=256 --loss1=0.5 --d_model=256 --num_attn_heads=4 --num_layers=2 --seed=42    --final_fc_dim=512 --n_blocks=4 --start=50 --learning_rate=0.0001  --dataset_name=assist2009  --use_CL=1 --cl_weight=0.02 --use_uncertainty_aug=1> output/assist2009/ukt_train &
```

## Evaluate

Now, let’s use `wandb_predict.py` to evaluate the model performance on the testing set.

```
python wandb_predict.py --save_dir=saved_model/YourModelPath
```

--save_dir is the save path of your trained model that you can find in your training log
