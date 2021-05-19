# ClassificationX

### Introduction
ClassificaitonX is an open source 2D image classfication toolbox based on Tensorflow2.4+.


### Changelog
- 2021.05.19 add resnest, shufflentv1, shufflenetv2.
- 2021.01.05 Initialized the repository.

### Model
#### ResNet
|    model   |  #Params   |   BFLOPs  | top-1 error | top-5 error |
| :--------: | :-------:  | :-------: | :---------: | :---------: |
|  ResNet18  |            |           |    30.24    |    10.92    |
|  ResNet34  |            |           |    26.70    |     8.58    |
|  ResNet50  |  22.73M    |   9.74    |    23.85    |     7.13    |
|  ResNet101 |            |           |    22.63    |     6.44    |
|  ResNet152 |  27.21M    |   22.6    |    21.69    |     5.94    |

#### Res2Net

|       model         |   #Params    |   BFLOPs  | top-1 error | top-5 error | link |
| :-----------------: | :----------: | :-------: | :---------: | :---------: | :---:|
|  Res2Net-50-48w-2s  |    25.29M    |    4.2    |    22.68    |    6.47     |      |
|  Res2Net-50-26w-4s  |    25.70M    |    4.2    |    22.01    |    6.15     |      |
|  Res2Net-50-14w-8s  |    25.06M    |    4.2    |    21.86    |    6.14     |      |
|  Res2Net-50-26w-6s  |    37.05M    |    6.3    |    21.42    |    5.87     |      |
|  Res2Net-50-26w-8s  |    48.40M    |    8.3    |    20.80    |    5.63     |      |
|  Res2Net-101-26w-4s |    45.21M    |    8.1    |    20.81    |    5.57     |      |
|  Res2Net-v1b-50     |    25.72M    |    4.5    |    19.73    |    4.96     |      |
|  Res2Net-v1b-101    |    45.23M    |    8.3    |    18.77    |    4.64     |      |
|  Res2NeXt-50        |    24.67M    |    4.2    |    21.76    |    6.09     |      |
|  Res2Net-DLA-60     |    21.15M    |    4.2    |    21.53    |    5.80     |      |
|  Res2NeXt-DLA-60    |    17.33M    |    3.6    |    21.55    |    5.86     |      |

#### CrossStageParrialNetwork
|       model         |   #Params    |   BFLOPs  | top-1 error | top-5 error | link |
| :-----------------: | :----------: | :-------: | :---------: | :---------: | :---:|
|  Darknet-53         |    41.57M    |   18.57   |    22.8    |      6.2     |      |
|  CSPDarkNet-53      |    27.61M    |   13.07   |    22.8    |      6.4     |      |

#### CrossStageParrialNetwork
|    model    | #Params | BFLOPs | top-1 error | top-5 error | link                                                         |
| :---------: | :-----: | :----: | :---------: | :---------: | ------------------------------------------------------------ |
| ResNeSt-50  |    -    |   -    |    18.95    |    5.58     | [baidu](https://pan.baidu.com/s/1nz6e_vlsJFxiV6QUZ9lXQQ)[5fzh] |
| ResNeSt-101 |    -    |   -    |    17.18    |    3.69     | [baidu](https://pan.baidu.com/s/13KpJwYJanIxzZcpTEsVvvg)[utbi] |
| ResNeSt-200 |    -    |   -    |    16.16    |    3.14     | [baidu](https://pan.baidu.com/s/1cqcBsjADcY7PXv-LZ30AYw)[x2jc] |
| ResNeSt-269 |    -    |   -    |    15.57    |    3.02     | [baidu](https://pan.baidu.com/s/1bASbT9yVP6tD0H-WkDbsLw)[7bap] |

