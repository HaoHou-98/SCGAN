# Semi-Cycled Generative Adversarial Networks for Real-World Face Super-Resolution

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.11.0](https://img.shields.io/badge/pytorch-1.11.0-green.svg?style=plastic)


![image](./docs/Frame.png)
**Figure:** *Architecture of the Semi-Cycled Generative Adversarial Network (SCGAN) for unsupervised face super resolution.*

We establish two independent degradation branches in the forward and backward cycle-consistent reconstruction processes, respectively, while the two processes share the same restoration branch. Our Semi-Cycled Generative Adversarial Networks (SCGAN) is able to alleviate the adverse effects of the domain gap between the real-world LR face images and the synthetic LR ones, and to achieve accurate and robust face SR performance by the shared restoration branch regularized by both the forward and backward cycle-consistent learning processes.

> **Semi-Cycled Generative Adversarial Networks for Real-World Face Super-Resolution** <br>
> H Hou, X Hu, J Xu, Y Hou, B Wei, D Shen <br>
> **arXiv preprint arXiv:2205.03777**


[[Paper](https://arxiv.org/pdf/2205.03777.pdf)]
[[Project Page](https://github.com/HaoHou-98/SCGAN)]



## Installation

Clone this repo.
```bash
git clone https://github.com/HaoHou-98/SCGAN.git
cd SCGAN/
```

 Please install dependencies by
```bash
pip install -r requirements.txt
```


## Dataset Preparation

The prepared test set and trainning set can be directly downloaded [here](https://drive.google.com/file/d/1BeIhDoeLyvIkJQuzFtIvNIBD0VTzgDjd/view?usp=sharing). After unzipping, put the `imgs_test`, `imgs_train` and `reference` folders in the root directory.


## Super-resolving Images Using Pretrained Models

Once the dataset is prepared, the results be got using pretrained models.


1. Inference

    ```
    python test.py
    ```

2. The results are saved at `./test_results/`.
3. Caculate the FID score of the results with the reference images.

	```bash
    python FID.py
    ```


## Training New Models

To train the new model, you need to put your own high-resolution and low-resolution face images into `./data_train/HIGH` and `./data_train/LOW`, respectively, and then
```bash
python train.py
```
The models are saved at `./train/models`




## Other Models
Will be released soon.



## Citation
If you use this code for your research, please cite our papers.
```
@article{hou2022semi,
  title={Semi-Cycled Generative Adversarial Networks for Real-World Face Super-Resolution},
  author={Hou, Hao and Hu, Xiaotao and Xu, Jun and Hou, Yingkun and Wei, Benzheng and Shen, Dinggang},
  journal={arXiv preprint arXiv:2205.03777},
  year={2022}
}
```
The code is released for academic research use only.
