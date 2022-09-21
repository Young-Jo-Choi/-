# FCN (Fully Convolutional Network) for Semantic Segmentation

## Abstract
- arbitrary한 input size를 받고 그에 상응하는 크기의 output을 출력하도록 하였다.
- AlexNet, VGG, GoogLeNet과 같은 classification model을 fine-tuning해 해당 모델이 사전 학습한 representation 능력을 가져오도록 하였다.
- skip architecture라는 것을 정의해 의미(semantic) 정보와 위치(appearance) 정보를 결합하도록 하였다.
- PASCAL VOC, NYUDv2, SIFT Flow 등에 대해 당시 기준 sota를 달성하였고, 일반적인 이미지 추론은 1/5초도 걸리지 않는 성과를 보였다.


## Fully convolutional network

- 기존 CNN - encoder (feature extraction) + fully connected layer
- 하지만 fully connected layer를 쓰는 것은 위치 정보가 손실되고, 고정된 크기의 입력 이미지만을 받아야한다는 단점이 존재

### Adapting classifiers for dense prediction (Convolution)
- LeNet, AlexNet 등 분류모델은 고정된 크기의 input을 받아 위치정보와 관련없는 output을 출력한다. 이런 network의 fully connected layer는 convolutional 층의 고정되고 flatten된 dimension을 가지며 공간좌표를 무시한다.
- 다음 그림은 AlexNet의 원래 구조와 해당 논문에서 이를 어떻게 변형시켰는지를 보여준다.

![original_to_fcn_heatmap](https://user-images.githubusercontent.com/59189961/191484984-7ba6ca51-ced2-435c-b49e-2ad6b041fbf6.png)

- Convolutional network 이후 fully connected layer를 추가해 1d 데이터로 만들어 분류를 수행한다.
- FCN에서는 1d가 아닌 3d 데이터가 되도록 1 $\times$ 1 convolution으로 대체해 1 $\times$ 1 $\times$ 4096, 1 $\times$ 1 $\times$ 1000 출력을 만들어낸다. 여기서 1000은 예시로 든 AlexNet이 분류해내는 classes의 개수인데 픽셀들이 어느 클래스에 속하는지 정보를 담고 있다.

- $n$ 개의 label이 있다면 1 $\times$ 1 ${\times} n$ 으로 마지막 layer를 구성할 수 있고, 분류모델처럼 flatten할 필요가 없다. input image는 이러한 구조에 의해 arbitrary하게 입력될 수 있다. 더 큰 이미지가 입력됐다면 약간의 정보손실을 거칠 뿐 연산에는 무리가 없다.

이런 spatial한 output map은 semantic segmantation과 같은 dense한 문제에 대해 대처할 수 있다. 모든 cell에 대해 갖고 있는 ground truth에 대해 forward와 backward 모두 간단하게 진행되며 convolution 고유의 계산 효율성을 활용한다. AlexNet의 경우 해당 backward 연산의 경우 단일 이미지의 경우 2.4ms, fully convolution 10x10 출력맵의 ruddn 37ms이며, 이는 forawrd pass와 유사한 속도향상을 초래한다.

분류 네트워크를 fully convolution으로 재해석하면 모든 크기의 input에 대해 output map이 산출되지만, 출력 차원은 일반적으로 샘플링에 의해 감소하고 coarse한 정보를 담게 된다.
(receptive field의 pixel strice와 동일한 요소만큼의 크기로 줄임)

![figure3_2](https://user-images.githubusercontent.com/59189961/191516526-119ea728-2865-4f17-829d-c57528ee791e.png)


### Upsampling is backwards strided convolution (Deconvolution)

- FCN에서는 transposed convolution, backward convolution, deconvolution 등으로 불리는 upsampling 연산을 통해 원래 input image size에 맞게 크기를 조정한다.

![K-004](https://user-images.githubusercontent.com/59189961/191492377-c298fc9f-0795-4731-863b-0b635d7fbff8.jpg)

- coarse한 output과 dense한 pixel 사이를 이어주기 위해서는 interpolation을 수행해야한다. simple bilinear interpolation은 input cell과 output cell의 상대적인 위치에 의존하는 선형 맵에 의해 가장 근접한 4개의 입력으로부터 각 출력 $y_{ij}$을 계산한다. 이 bilinear interpolation은 deconvlutional layer를 초기화하는데 사용된다.
- deconvolution layer가 여러 개 쌓이면 nonlinear upsampling으로 기능할 수 있다.
- 하지만 이런 upsampling 연산은 매우 작은 spatial informations에서 크기를 늘리는 것이기 때문에 많은 정보의 손실을 유발할 수 있다. 이미 convolution part를 지나며 잃어버린 정보도 상당할 수 있다. 
- 가장 coarse하게 압축된 의미(semantic) 정보로부터 보간된 것이기 때문에 위치(spatial) 정보를 잃지 않기위한 조치가 필요하다.


## 4. Segmentation Architecture

### From classifier to dense FCN
- 본 논문 실험에서는 VGG 16-layer net을 사용 (AlexNet, VGG, GooLeNet을 모두 사용해 실험했는데 VGG가 sota를 달성했기 때문)
- 마지막 classifying하는 fully connected layer를 1 $\times$ 1 convolutional layer with channel dimension 21로 변경, 해당 layer로 coarse한 출력을 얻는다.
- deconvolutional layer를 추가해 pixel 단위의 예측으로 bilinearly upsampling한다.

### Combining what and where
- Fully Convolutional net의 구조를 다음과 같이 만들어 출력의 spatial precision을 개선하였다.

![figure3](https://user-images.githubusercontent.com/59189961/190954547-d974a646-90e1-4ef1-a039-dafc1cb1e8a1.png)

해당 구조의 예시에서 convolution layer들을 지나며 $\div$ 2 씩 pixel의 양이 줄어든다. pool5 층을 지난 데이터를 직접적으로 32 $\times$ upsampling하여 예측을 수행할 수 있는데 이를 FCN-32s라고 한다.

하지만 이런 예측은 너무 많은 위치 정보를 손실한 coarest 상태에서 추가적인 정보 없이 바로 예측을 수행하였기 때문에 결과가 매우 좋지 않으리라 생각할 수 있을 것이다.

더 이전 계층으로 돌아갈 수록 original 정보를 더 많이 갖고 있을 것이기 때문에 이 부분을 결합해 모델 예측에 영향을 미치도록 (leverage) 하는 방법을 생각해 볼 수 있다.
pool5 계층의 출력에 2 $\times$ upsample만을 취한 후 pool4의 출력과 결합해 예측을 수행하면 위치 정보가 조금 더 보존된 상태일 것이라 생각해볼 수 있다. 이때의 모델을 FCN-16s라고 한다.

마찬가지로 16 $\times$ upsampling 직전의 데이터를 다시 2 $\times$ upsampling하여 pool3 계층의 출력과 결합한 후 8 $\times$ upsampling하여 예측에 사용한 모델을 FCN-8s라고 한다.
이런 구조를 final prediction layer를 좀더 미세한 stride를 갖는 lower layer와 결합하는 skip connection라 하며 아래 그림을 통해 각각의 결과를 확인할 수 있다.

![figure4](https://user-images.githubusercontent.com/59189961/190954570-d019d9d3-8456-4828-8251-d76d886251eb.png)

- 더 previous한 층과 skip connection할 수록 위치정보까지 포함해 잘 예측하는 것을 확인할 수 있다.
- fine layer와 coarse layer를 결합하면 각 픽셀 단위의 예측과 더불어서 위치 정보에 대한 예측을 함께 하도록 만들 수 있다.


![result](https://user-images.githubusercontent.com/59189961/190954663-34770038-d507-47a2-9a97-82fc1239aaa0.png)


(PASCAL VOC의 데이터셋 구조와 해당 데이터를 이용한 semantic segmentation 모델의 학습은 간단한 코드로 올려두었으니 참조바람)
