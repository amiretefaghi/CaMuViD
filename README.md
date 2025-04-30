# CaMuViD: Calibration-Free Multi-View Detection

The official implementation of

[CaMuViD: Calibration-Free Multi-View Detection](https://amiretefaghi.github.io/Camuvid.html).

## Introduction

CaMuViD is an advanced multi-view pedestrian detection model developed by researchers from AI Laboratory, University of Florida. The backbone of this model is based on [InternImage](https://github.com/OpenGVLab/InternImage), InternImage employs DCNv3 as its core operator. This approach equips the model with dynamic and effective projection and back projection estimation networks to transform from image space feature maps to shared space feature maps for multi-views.

<div align=center>
<img src='./docs/figs/intro.png' width=400>
</div>


## Performance

- InternImage achieved an impressive Top-1 accuracy of 90.1% on the ImageNet benchmark dataset using only publicly available data for image classification. Apart from two undisclosed models trained with additional datasets by Google and Microsoft, InternImage is the only open-source model that achieves a Top-1 accuracy of over 90.0%, and it is also the largest model in scale worldwide.
- InternImage outperformed all other models worldwide on the COCO object detection benchmark dataset with a remarkable mAP of 65.5, making it the only model that surpasses 65 mAP in the world.
- InternImage also demonstrated world's best performance on 16 other important visual benchmark datasets, covering a wide range of tasks such as classification, detection, and segmentation, making it the top-performing model across multiple domains.

**Multi-View Detection Results**

| Method                   | WT MODA ↑ | WT MODP ↑ | WT Prec. ↑ | WT Rec. ↑ | WT F1 ↑ | MVX MODA ↑ | MVX MODP ↑ | MVX Prec. ↑ | MVX Rec. ↑ | MVX F1 ↑ |
|--------------------------|:---------:|:---------:|:----------:|:---------:|:-------:|:----------:|:----------:|:-----------:|:---------:|:--------:|
| RCNN & clustering        |   11.3    |   18.4    |   68.0     |   43.0    |  52.7   |   18.7*    |   46.4*    |    63.5*    |   43.9*   |  51.9    |
| POM-CNN                  |   23.2    |   30.5    |   75.0     |   55.0    |  63.5   |     –      |     –      |      –      |     –     |    –     |
| DeepMCD                  |   67.8    |   64.2    |   85.0     |   82.0    |  83.5   |   70.0*    |   73.0*    |    85.7*    |   83.3*   |  84.5    |
| Deep-Occlusion           |   74.1    |   53.8    |   95.0     |   80.0    |  86.8   |   75.2*    |   54.7*    |    97.8*    |   80.2*   |  88.1    |
| MVDet                    |   88.2    |   75.7    |   94.7     |   93.6    |  94.1   |   83.9     |   79.6     |    96.8     |   86.7    |  91.5    |
| SHOT                     |   90.2    |   76.5    |   96.1     |   94.0    |  95.0   |   88.3     |   82.0     |    96.6     |   91.5    |  94.0    |
| MVDeTr                   |   91.5    |   82.1    |   97.4     |   94.0    |  95.7   |   93.7     |   91.3     |    99.5     |   94.5    |  96.9    |
| MVAug                    |   93.2    |   79.8    |   96.3     |   97.0    |  96.6   |   95.3     |   89.7     |    99.4     |   95.9    |  97.6    |
| 3DROM                    |   93.5    |   75.9    |   97.2     |   96.2    |  96.7   |   95.0     |   84.9     |    99.0     |   96.1    |  97.5    |
| MVFP                     |   94.1    |   78.8    |   96.4     |   97.7    |  97.0   |   95.7     |   82.1     |    98.4     |   97.2    |  97.8    |
| TrackTacular             |   93.2    |   77.5    |   97.3     |   95.8    |  96.5   |   96.5     |   75.0     |    99.4     |   97.1    |  98.2    |
| **Ours**                 | **95.0**  | **80.9**  | **96.3**   | **98.6**  |**97.4** | **96.5**   | **89.3**   | **97.9**    | **98.6**  |**98.3**  |

\* reported by original authors

---

**Cross-Dataset Performance**

| Method         | MODA | MODP  | Prec. | Recall | F1   |
|----------------|:----:|:-----:|:-----:|:------:|:----:|
| MVDet [17]     | 17.0 | 65.8  | 60.5  | 48.8   | 54.0 |
| MVAug [10]     | 26.3 | 58.0  | 71.9  | 50.8   | 59.5 |
| MVDeTr [16]    | 50.2 | 69.1  | 74.0  | 77.3   | 75.6 |
| SHOT [34]      | 53.6 | 72.0  | 75.2  | 79.8   | 77.4 |
| GMVD [39]      | 66.1 | 72.2  | 82.0  | 84.7   | 83.3 |
| 3DROM [30]     | 67.5 | 65.6  | 94.5  | 71.7   | 81.5 |
| MVFP [1]       | 76.7 | 74.9  | 85.2  | 92.8   | 88.8 |
| **Ours**       |**86.4**|**60.7**|**89.3**|**98.1**|**93.5**|

---

**Camera-Elimination Ablation**

| C1 | C2 | C3 | C4 | C5 | C6 | C7 | MODA  | MODP | Precision | Recall |  F1  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:-----:|:----:|:---------:|:------:|:----:|
| ✓  | ✗  | ✗  | ✗  | ✗  | ✗  | ✗  | 60.1  | 65.0 |   99.8    | 60.2   | 75.1 |
| ✓  | ✓  | ✗  | ✗  | ✗  | ✗  | ✗  | 77.8  | 71.5 |   99.8    | 77.9   | 87.5 |
| ✓  | ✓  | ✓  | ✗  | ✗  | ✗  | ✗  | 90.6  | 76.4 |   98.4    | 92.1   | 95.1 |
| ✓  | ✓  | ✓  | ✓  | ✗  | ✗  | ✗  | 93.8  | 79.4 |   98.3    | 95.5   | 96.9 |
| ✓  | ✓  | ✓  | ✓  | ✓  | ✗  | ✗  | 93.8  | 80.0 |   96.8    | 96.9   | 97.0 |
| ✓  | ✓  | ✓  | ✓  | ✓  | ✓  | ✗  | **95.6** | 80.2 |   96.6    | 99.3   | 98.0 |
| ✓  | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  | 95.0  | 80.9 |   96.3    | 98.6   | 97.4 |


## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@inproceedings{etefaghi2025camuvid,
  title={CaMuViD: Calibration-Free Multi-View Detection},
  author={Amir Etefaghi Daryani, M. Usman Maqbool Bhutta, Byron Hernandez, Henry Medeiros},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```