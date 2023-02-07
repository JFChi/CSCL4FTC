# CSCL4FTC

Official Implementation of EMNLP findings 2022 paper [Conditional Supervised Contrastive Learning for Fair Text Classification](https://arxiv.org/abs/2205.11485).

# Reference

If you find our code useful, please cite

```
@inproceedings{chi-etal-2022-conditional,
    title = "Conditional Supervised Contrastive Learning for Fair Text Classification",
    author = "Chi, Jianfeng  and
      Shand, William  and
      Yu, Yaodong  and
      Chang, Kai-Wei  and
      Zhao, Han  and
      Tian, Yuan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.199",
    pages = "2736--2756",
    abstract = "Contrastive representation learning has gained much attention due to its superior performance in learning representations from both image and sequential data. However, the learned representations could potentially lead to performance disparities in downstream tasks, such as increased silencing of underrepresented groups in toxicity comment classification. In light of this challenge, in this work, we study learning fair representations that satisfy a notion of fairness known as equalized odds for text classification via contrastive learning. Specifically, we first theoretically analyze the connections between learning representations with a fairness constraint and conditional supervised contrastive objectives, and then propose to use conditional supervised contrastive objectives to learn fair representations for text classification. We conduct experiments on two text datasets to demonstrate the effectiveness of our approaches in balancing the trade-offs between task performance and bias mitigation among existing baselines for text classification. Furthermore, we also show that the proposed methods are stable in different hyperparameter settings.",
}
```

## Installation

We test our implementation in Python 3.7-3.9. Please install the library via the following command.
```
pip install -r requirements.txt
```

## Data Preprocessing

```
pip install gdown
mkdir -p data && cd data

# download jigsaw data
gdown --no-check-certificate --folder 1v0QzeIgVbjdxCv4EDz4Joo4lGH4MKBom

# download biasbios data
gdown --no-check-certificate --folder 1QuJryP0DspGLp8ivXdA5JMNevtNote7p
```

Or you can download the dataset in the google drive via this [link](https://drive.google.com/drive/folders/1LYK4oPVp58NteKjm-p_8IgKnAeduXbnQ?usp=sharing).

## Running Contrastive Learning for Fair Text Classifcation

### Data Augmentation
Data augmentation is a key part for CL. The following commands show how we perform EDA on jigsaw dataset:
```
# idx_end is the last example index to perform data augmentation 
# and we can set it to be larger than number of examples in the training split
python augment_data.py --dataset jigsaw-race --split train --idx_end 30000 --aug_type EDA 
python augment_data.py --dataset jigsaw-race --split val --idx_end 30000 --aug_type EDA
python augment_data.py --dataset jigsaw-race --split test --idx_end 30000 --aug_type EDA

python format_data_augmentation.py --dataset jigsaw-race --split train --aug_type EDA
python format_data_augmentation.py --dataset jigsaw-race --split val --aug_type EDA
python format_data_augmentation.py --dataset jigsaw-race --split test --aug_type EDA
```

### Running Two-stage CL

The following command runs two-stage CL with hyperparameters lambda=1.0 (weight of CS-infoNCE loss), batch size=64, temperature=2.0
```
# run 
gpu_id=0
aux_weight=1.0
seed=1234
MODEL_PATH="results/EMNLP22/jigsaw-race/CL+CE/weight_${aux_weight}/seed_${seed}"
./sh/run_gradcache_pretrain_contrastive_jigsaw_race.sh -b 64 -t 2.0 -c "--aug_type EDA" -g $gpu_id -s $seed -w $aux_weight -p $MODEL_PATH && ./sh/run_finetune_CE_jigsaw_race.sh -g $gpu_id -p $MODEL_PATH -s $seed
```

### Running One-stage CL
The following command runs two-stage CL with hyperparameters lambda=1.0 (weight of CS-infoNCE loss), batch size=120, temperature=2.0, scl_ratio=0.3
```
gpu_id=0
aux_weight=1.0
seed=1234
MODEL_PATH="results/EMNLP22/jigsaw-race/one-stage_CL/weight_${aux_weight}/seed_${seed}"
./sh/run_one_stage_cl_jigsaw_race.sh -b 128 -t 2.0 -r 0.3 -c "--aug_type EDA" -p $MODEL_PATH -g $gpu_id -s $seed -w $aux_weight
seed=1234
```

## Running the Baselines

### CE training
Run classifcation using cross-entropy loss
```
gpu_id=0
seed=1234
OUT_PATH="results/EMNLP22/jigsaw-race/end-to-end_CE/seed_${seed}"
./sh/run_train_CE_jigsaw_race.sh -g $gpu_id -o $OUT_PATH -s $seed 
```

### INLP
Running INLP involves two steps: (1) first we encode CE training representation; (2) we learn the INLP linear layer and classifier.
```
# encode CE training representation 
dataset=jigsaw-race
type=best
model_name=CE_trained_bert
seed=1234
for split in train val test
do
    model_path=results/EMNLP22/$dataset/end-to-end_CE/seed_${seed}/$type
    out_dir=INLP/data/EMNLP22/$dataset/end-to-end_CE/seed_${seed}/$type
    ./sh/INLP/run_encode_bert_states.sh -d $dataset -l $split -p $model_path -n $model_name -o $out_dir
done

# train INLP
gpu_id=0
dataset=jigsaw-race
type=best
model_name=CE_trained_bert
num_clfs=20
seed=1234
encoded_data_path=INLP/data/EMNLP22/$dataset/end-to-end_CE/seed_${seed}/$type
output_dir=results/EMNLP22/$dataset/INLP/CE_pretrained+pt_finetune_$type/num_clfs_${num_clfs}/seed_${seed}/
./sh/INLP/run_INLP_pt_jigsaw_race.sh -s $seed -p $encoded_data_path -n $model_name -l $num_clfs -o $output_dir -g $gpu_id
```

### Adversarial Training
```
# running single adversary with lambda=0.1
gpu_id=0
lambda_adv=0.1
n_disciminators=1
seed=1234
OUT_PATH="results/EMNLP22/jigsaw-race/adv_training_by_class/N_${n_disciminators}_lambda_adv_${lambda_adv}/seed_${seed}"
./sh/diverse_adv_training/run_adv_train_jigsaw_race.sh -g $gpu_id -s $seed -n $n_disciminators -l $lambda_adv -r 0.0 -o $OUT_PATH -c "--by_class" && rm -r $OUT_PATH/tmp/

# running multiple adversary with lamdda=0.1
gpu_id=0
lambda_adv=0.1
lambda_diff=100
n_disciminators=3
seed=1234
OUT_PATH="results/EMNLP22/jigsaw-race/adv_training_by_class/N_${n_disciminators}_lambda_adv_${lambda_adv}_diff_${lambda_diff}/seed_${seed}"
./sh/diverse_adv_training/run_adv_train_jigsaw_race.sh -g $gpu_id -s $seed -n $n_disciminators -l $lambda_adv -r $lambda_diff -o $OUT_PATH -c "--by_class" && rm -r $OUT_PATH/tmp/
```
