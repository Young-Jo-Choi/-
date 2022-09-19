# FCN (Fully Convolutional Network) for Semantic Segmentation

## 1. Abstract
- arbitrary한 input size를 받고 그에 상응하는 크기의 output을 출력하도록 하였음
- AlexNet, VGG, GoogLeNet과 같은 classification model을 fine-tuning해 해당 모델이 사전 학습한 representation 능력을 가져오도록 하였음
- skip architecture라는 것을 정의해 의미(semantic) 정보와 위치(appearance) 정보를 결합하도록 하였음
- PASCAL VOC, NYUDv2, SIFT Flow 등에 대해 당시 기준 sota를 달성하였고, 일반적인 이미지 추론은 1/5초도 걸리지 않는 성과를 보임

## 2. Introduction

(patchwise가 아니라는 내용 추가 필요 -> upsampling 부분 포함)
(Fully convolutional versions of existing networks predict dense outputs from arbitrary-sized inputs. Both learning and inference are performed whole-image-ata-time by dense feedforward computation and backpropagation. In-network upsampling layers enable pixelwise prediction and learning in nets with subsampled pooling.)

(intorduction은 생략해도 될지도??)

## 3. Fully convolutional network

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

$$f_{ks} \circ g_{k\prime s\prime} = (f \circ g)_{k\prime + (k-1){s\prime},s{s\prime}} $$

(어쩌구 저쩌구)

### 3.1 Adapting classifiers for dense prediction
- LeNet, AlexNet 등 분류모델은 고정된 크기의 input을 받아 위치정보와 관련없는 output을 출력한다. 이런 network의 fully connected layer는 고정된 dimension을 가지며 공간좌표를 무시한다.
- 이런 fully connected layer는 전체 영역을 커널로 갖는 convolution으로 볼 수도 있다.
(figure2 삽입)

Furthermore, while the resulting maps are equivalent to the evaluation of the original net on particular input patches, the computation is highly amortized over the overlapping regions of those patches. For example, while AlexNet takes 1.2 ms (on a typical GPU) to infer the classification scores of a 227×227 image, the fully convolutional net takes 22 ms to produce a 10×10 grid of outputs from a 500×500 image, which is more than 5 times faster than the na¨ıve approach.

이런 spatial한 output map은 semantic segmantation과 같은 dense한 문제에 대해 대처할 수 있다. 모든 cell에 대해 갖고 있는 ground truth에 대해 forward와 backward 모두 간단하게 진행되며 convolution 고유의 계산 효율성을 활용한다. AlexNet의 경우 해당 backward 연산의 경우 단일 이미지의 경우 2.4ms, fully convolution 10x10 출력맵의 ruddn 37ms이며, 이는 forawrd pass와 유사한 속도향상을 초래한다.(위 번역안된 부분에 이어지는 내용임)

분류 네트워크를 fullt convolution으로 재해석하면 모든 크기의 input에 대해 output map이 산출되지만, 출력 차원은 일반적으로 샘플링에 의해 감소하고 coarse한 정보를 담게 된다.(receptive field의 pixel strice와 동일한 요소만큼의 크기로 줄임)

### 3.2 Shift-and-strich is filter rarefaction


### 3.3 Upsampling is backwards strided convolution
- coarse한 output과 dense한 pixel 사이를 이어주기 위해서는 interpolation을 수행해야한다. simple bilinear interpolation은 input cell과 output cell의 상대적인 위치에 의존하는 선형 맵에 의해 가장 근접한 4개의 입력으로부터 각 출력 $y_{ij}$을 계산한다.

- transposed convolution, backward convolution, deconvolution
- 코드에 들어있는 그림 참조
- deconvolution layer가 여러개 쌓이면 nonlinear upsampling으로 기능할 수 있다.

### 3.4 Patchwise training is loss sampling

## 4. Segmentation Architecture
(figure3 삽입)
### 4.1 From classifier to dense FCN

### 4.2 Combining what and where


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

