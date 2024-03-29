# IEEE-TMI: Graph-based Region and Boundary Aggregation for Biomedical Image Segmentation

# ODOC Segmentation


# Data Preparation  
```
Prepare your data, the index of the test data is in test.list  
```  
# Pre-trained Model  

--Download [best_model.pth](https://drive.google.com/file/d/1S7s4jq8emUQbDHoG7_VUMcBWpsdV7gaR/view?usp=sharing) and put it into ./model/  

# Pred-trained backbone  

--Download [res2net50_v1b_26w_4s-3cf99910.pth](https://drive.google.com/file/d/1FLMVNCRFJGlMN65r8cVsopUZbhB6u83I/view?usp=sharing) and put it into ./res_weight/ 


# Predict  
```
 -- Run test.py  
 ```  

# Citation
If you find our work useful or our work gives you any insights, please cite:
```
@article{meng2021graph,
  title={Graph-based region and boundary aggregation for biomedical image segmentation},
  author={Meng, Yanda and Zhang, Hongrun and Zhao, Yitian and Yang, Xiaoyun and Qiao, Yihong and MacCormick, Ian JC and Huang, Xiaowei and Zheng, Yalin},
  journal={IEEE transactions on medical imaging},
  volume={41},
  number={3},
  pages={690--701},
  year={2021},
  publisher={IEEE}
}

```

# Acknowledgement
 Part of our code is built based on [PraNet](https://github.com/DengPingFan/PraNet), thanks Dr.Fan Dengping for such a good project.

 






