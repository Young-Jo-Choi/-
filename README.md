# FCN (Fully Convolutional Network) for Semantic Segmentation

## Abstract
- arbitrary한 input size를 받고 그에 상응하는 크기의 output을 출력하도록 하였음
- AlexNet, VGG, GoogLeNet과 같은 classification model을 fine-tuning해 해당 모델이 사전 학습한 representation 능력을 가져오도록 하였음
- skip architecture라는 것을 정의해 의미(semantic) 정보와 위치(appearance) 정보를 결합하도록 하였음
- PASCAL VOC, NYUDv2, SIFT Flow 등에 대해 당시 기준 sota를 달성하였고, 일반적인 이미지 추론은 1/5초도 걸리지 않는 성과를 보임

## Introduction

(patchwise가 아니라는 내용 추가 필요 -> upsampling 부분 포함)
(Fully convolutional versions of existing networks predict dense outputs from arbitrary-sized inputs. Both learning and inference are performed whole-image-ata-time by dense feedforward computation and backpropagation. In-network upsampling layers enable pixelwise prediction and learning in nets with subsampled pooling.)

(intorduction은 생략해도 될지도??)

## Fully convolutional network

기존 CNN - encoder (feature extraction) + fully connected layer
(개고양이 분류 구조 사진 첨부)
하지만 fully connected layer를 쓰는 것은 위치 정보가 손실되고, 고정된 크기의 입력 이미지만을 받아야한다는 단점이 존재
fc layer의 한계 - 위치 정보의 손실, 입력 이미지 크기가 고정됨
higher layer의 위치 정보는 path-connected 되어 있는 image 내의 위치와 correspond하는데 이를 receptive fields라고 한다.

$$y_{ij} = f_{ks}(\\{x_{si}+δ_{i,sj+δj}\\}_ {0≤δ_{i},δ_{j}≤k})$$

$$x_{ij} : \text{data vector at location (i,j)}$$
$$y_{ij} : \text{location (i,j) of following layer}$$
$$k : \text{kernel size}$$
$$s : \text{stride or subsampling factor}$$
$f_{ks}$는 layer의 type을 결정하는 함수이다.


어떤 방법을 사용했는지 등등
기존 연구와의 차이점은 무엇인지

출력층을 fully connected layer --> Convnet으로 변경(모양은?)
위치정보가 매우 중요
(개,고양이 사진 첨부)

(fc layer->convnet 사진 첨부)

convnet으로 변하는 출력층의 shape 설명
FCN의 출력은 coarse

Deconvolution
연산과정은 d2l 책 참고

skip connection
figure3 첨부 
의미에 대해 설명

결과
figure4, result_table

