### Usage For Running With Python

#### Packages

#### **Cuda Home To Export**

​		export CUDA_HOME = YourAnacondaPath/anaconda3/pkgs/cudatoolkit-your_vesion_to_replace

​    eg : export CUDA_HOME=/new_disk_B/dyx/anaconda3/pkgs/cudatoolkit-10.1.243-h6bb024c_0/

#### **Eval**

​        python eval.py --cfg experiments/vgg16_pgnn_structure_willow.yaml --epoch 6

#### **Train**

​        python train_eval.py --cfg experiments/vgg16_pgnn_structure_willow.yaml
​        python train_eval.py --cfg experiments/vgg16_pgnn_structure_voc.yaml
​        python train_eval.py --cfg experiments/vgg16_pgnn_structure_cub.yaml
​        python train_eval.py --cfg experiments/vgg16_pgnn_structure_imcpt.yaml

#### **Dataset**

​        Here we release four datasets in our experiments.  The link of downloading datasets is as follows: https://drive.google.com/file/d/1crw9dXAlC_Qd9Z12qsEb3Sq0YmLnpaTt/view?usp=sharing.

#### **Device**

​        If you want to change device number on GPU, please modify "Motif_Position/utils_pgnn.py" 's device number.

#### **Iteration**

​		Use iteration=True in "PGNN_structure/model.py" to see how iterative position and structure GCN works.

#### **Cite**

​        If you want to use our implementations, please cite as follow:

