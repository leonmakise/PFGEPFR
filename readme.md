# CVPR2021 'Pseudo Facial Generation with Extreme Poses for Face Recognition'


# Table Of Contents
-  [Requirements](#requirements)
-  [Experiments on MultiPIE dataset](#experiments-on-multipie-dataset)
<!-- -  [Experiments on CFP dataset](#experiments-on-cfp-dataset)
-  [Experiments on LFW dataset](#experiments-on-lfw-dataset)
-  [Experiments on IJB dataset](#experiments-on-ijb-dataset)
-  [Experiments on MegaFace dataset](#experiments-on-megaface-dataset)
-  [Future Work](#future-work)
-  [Acknowledgments](#acknowledgments) -->

# Requirements
## For finetuning LightCNN
Considering that the original LightCNN_V2is based on an old environment, so we need create one based on CUDA9.2. 
  - cuda92
  - numpy
  - opencv
  - python=2.7
  - pytorch=0.4.1
  - scikit-learn
  - scipy
  - torchvision=0.2.1
  - tqdm

## For training the generator
  - cudatoolkit=10.0
  - numpy
  - opencv
  - python=3.6
  - pytorch=1.4.0
  - scikit-learn
  - scipy
  - torchvision=0.5.0
  - tqdm

**You can also try some other versions (Most of GPUs are now Nvidia 30s, and they do not support CUDA lower than 11.0), but you may encounter some dependency problems which can be easily fixed by changing some out-of-date functions.**

# Datasets
## [MultiPIE](https://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html)
<!-- ## [CFP](http://www.cfpw.io)
## [LFW](http://vis-www.cs.umass.edu/lfw/)
## [IJB-B&IJB-C](https://nigos.nist.gov/datasets/ijbc/request)
## [MegaFace](http://megaface.cs.washington.edu) -->


# Experiments on MultiPIE dataset  
Due to the copyright, we can only offer a script to test the pretrained models easily, also you can train models from scratch if you want. So **for example** assume you want to test it when the angle is 90 $\degree$ in our setting2, you should do the following:


- In `experiment-on-multipie/pre_train`  folder, you should put the LightCNN_V2 model which is named as `lightCNN_152_checkpoint.pth.tar`. In `experiment-on-multipie/model` folder, you should put the generator model which is named as `netG_model_epoch_80_iter_0.pth`. 

<!-- ```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
```  -->

   
- Then, you should go into `experiment-on-multipie/run_evaluation.sh` and check the path of data and model. Depending on your device, you can adjust the batch size and GpuID as you like. `--probe_list` depend what angle you are testing. In this example, we should choose `multipie_90_test_list.txt` in the MultiPIE data folder.

<!-- ```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
``` -->

- Finally, you just run the script we offer:
```python
sh run_evaluation.sh 80
``` 
 And the results should be like this:

![](.\MultiPIE_setting2.png)

**For any other angels, you can test them easily by changing the `probe_list` file. For setting1, you can change the data file to.**


<!-- # In Details
```
├──  config
│    └── defaults.py  - here's the default config file.
│
│
├──  configs  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│
│
├── modeling            - this folder contains any model of your project.
│   └── example_model.py
│
│
├── solver             - this folder contains optimizer of your project.
│   └── build.py
│   └── lr_scheduler.py
│   
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
│ 
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_you_need
│ 
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
``` -->













# Citation
If you find this project useful in your research, please consider citing:

@inproceedings{DBLP:conf/cvpr/WangM0L021,
  author    = {Guoli Wang and
               Jiaqi Ma and
               Qian Zhang and
               Jiwen Lu and
               Jie Zhou},
  title     = {Pseudo Facial Generation With Extreme Poses for Face Recognition},
  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}
               2021, virtual, June 19-25, 2021},
  pages     = {1994--2003},
  year      = {2021},
}
# Future Work
More exciting researches are under constructions. We will release them later.

# Contacts
If you have any questions, you can send email to wangguoli1990@mail.tsinghua.edu.cn and jiaqima@whu.edu.cn.


<!-- # Acknowledgments -->


