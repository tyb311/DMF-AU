# DMF-AU
PyTorch implementation for our paper (Accepted by Computers in Biology and Medicine):    

"A Lightweight Network Guided with Differential Matched Filtering for Retinal Vessel Segmentation"


## Project

/data:dataset & dataloader & data pre-processing

/nets:implementation for our network (DMF-AU)

/utils:implementation for loss function & optimizer

/onnx
-   Pytorch trained weights from DRIVE, STARE datasets.
-   The *.onnx weights can be directly used to extract vessels from fundus images, see onnx\infer.py

/results
-   segmentation results for popular datasets
-   segmentation results for cross-dataset-validation


## Contact
For any questions, please contact me. 
And my e-mails are 
-   tyb311@qq.com
-   tyb@std.uestc.edu.cn

```BibTex
@article{tan2023lightweight,
  title={A lightweight network guided with differential matched filtering for retinal vessel segmentation},
  author={Tan, Yubo and Zhao, Shi-Xuan and Yang, Kai-Fu and Li, Yong-Jie},
  journal={Computers in Biology and Medicine},
  pages={106924},
  year={2023},
  publisher={Elsevier}
}
```
