# Hierarchical Graph Learning with Small-World Brain Connectomes for Cognitive Prediction [<a href="https://link.springer.com/chapter/10.1007/978-3-031-72086-4_29">paper</a>]

[python-img]: https://img.shields.io/github/languages/top/ZhihaoPENG-CityU/MM21---AGCN?color=lightgrey
[![Made with Python][python-img]]


Our four brain-oriented submitted papers are all accepted by MICCAI 2024 (one Early Accept)
# Abstract
*Functional MRI is capable of assessing an individual’s cognitive ability by blood oxygen level dependence. Due to the complexity of brain function, exploring the relationship between cognitive ability and brain functional connectivity is extremely challenging. Recently, graph neural networks have been employed to extract functional connectivity features for predicting cognitive scores. Nevertheless, these methods have two main limitations: 1) Ignore the hierarchical nature of brain: discarding fine-grained information within each brain region, and overlooking supplementary information on the functional hierarchy of the brain at multiple scales; 2) Ignore the small-world nature of brain: current methods for generating functional connectivity produce regular networks with relatively low information transmission efficiency. To address these issues, we propose a Hierarchical Graph Learning with Small-World Brain Connectomes (SW-HGL) framework for cognitive prediction. This framework consists of three modules: the pyramid information extraction module (PIE), the small-world brain connectomes construction module (SW-BCC), and the hierarchical graph learning module (HGL). Specifically, PIE identifies representative vertices at both micro-scale (community level) and macro-scale (region level) through community clustering and graph pooling. SW-BCC simulates the small-world nature of brain by rewiring regular networks and establishes functional connections at both region and community levels. MSFEF is a dual-branch network used to extract and fuse micro-scale and macro-scale features for cognitive score prediction. Compared to state-of-the-art methods, our SW-HGL consistently achieves outstanding performance on HCP dataset.*

We appreciate it if you use this code and cite our related papers, which can be cited as follows,

> @inproceedings{jiang2024HGL, <br>
>   title = {Hierarchical Graph Learning with Small-World Brain Connectomes for Cognitive Prediction}, <br>
>   author = {Jiang, Yu, He, Zhibin, Peng, Zhihao, Yuan, Yixuan.}, <br>
>   booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention}, <br>
>   pages =【306--316}, <br>
>   isbn = {978-3-031-72086-4}, <br>
>   year = {2024}
> } <br>


Jiang, Y., He, Z., Peng, Z., Yuan, Y. (2024). Hierarchical Graph Learning with Small-World Brain Connectomes for Cognitive Prediction. In: Linguraru, M.G., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. MICCAI 2024. Lecture Notes in Computer Science, vol 15005. Springer, Cham. https://doi.org/10.1007/978-3-031-72086-4_29

# Environment
+ Python: 3.10.13
+ Pytorch: 2.10.3
+ CUDA Version: 12.1
+ NVIDIA GeForce RTX 4090


# To run code
First, the downloaded original nii format fMRI is divided into three sets: train, val and test according to the ratio of 7:2:1, and placed in the corresponding folder of dataset. Use data_preprocess to preprocess the data that can be input into the network, and the obtained data is stored in the respective input folders. Use 

python 1-data_prepocess.py

Then we start the cognitive prediction task. Use
python 2-train.py

