# An End-to-end Knowledge-driven Multi-instance Learning Network with Hierarchical Cluster-incorporated Aware Filtering for Larynx Pathological Grading 

## :fire: News

- [2025/04/25] Our manuscript was currently Under Review  _IEEE Journal of Biomedical and Health Informatics_.



## :rocket: Pipeline

Here's an overview of our **Hierarchical Cluster-incorporated Aware Filtering (HCF-MIL)** method:

![Figure 1](./images/main.jpg)



## :mag: TODO
<font color="red">**We are currently organizing all the code. Stay tuned!**</font>
- [x] training code
- [x] Evaluation code
- [x] Model code
- [ ] Pretrained weights
- [ ] Datasets





## 🛠️ Getting Started

To get started with NCFM, follow the installation instructions below.

1.  Clone the repo

```sh
git clone https://github.com/Prince-Lee-PathAI/HCF-MIL
```

2. Install dependencies
   
```sh
pip install -r requirements.txt
```

3. Training on Swin Transformer-S Backbone
```sh
sh run_swinT.sh
Modify: --abla_type sota --run_mode train --random_seed ${seed}
```

4. Evaluation
```sh
sh run_swinT.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed}
```

5. Extract features for plots
```sh
sh run_swinT.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --feat_extract
```

6. Interpretability plots
```sh
sh run_swinT.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --bag_weight
```

## :postbox: Contact
If you have any questions, please contact [Chentao Li](https://prince-lee-pathai.github.io/) (`cl4691@columbia.edu`).
