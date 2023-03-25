# Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
논문 : https://arxiv.org/pdf/1610.02391.pdf

## CNN
(이미지 : CNN)<br>
대충 설명<br>

## CAM
논문 : Learning Deep Features for Discriminative Localization.<br>
- 일반 CNN architecture의 output만으로는 분류 결과에 대한 모델이 어디를 주목하는지 알 수가 없다.
- 보통의 deep learning 구조는 내부 parameter를 일일히 확인하기 어려운 black box 구조이기 때문에 모델 해석의 관점에서 어려움이 따른다.

(이미지 : feature maps)<br>
<- low level feature maps | high level feature maps -><br>
- layer가 얕을 수록 low-level feature를 읽는다.
- layer가 깊어질 수록 조금 더 고차원적인 특징(class specific하도록 semantic한 정보)을 포착한다.
- 때문에 last layer의 activation map에 주목하여 model이 실제 주목하는 영역을 살펴볼 수 있다.

(이미지 : CAM)
- 대충 global average pooling (channel별로 하나의 값을 뽑아내기 위해)
- 대충 fintuning
- 대충 weight 획득

$$ 
\begin{align*}
S_c &= \sum_k w_k^c \sum_{x,y} f_{k}(x,y) \\
&= \sum_{x,y}\sum_k w_k^c f_{k}(x,y) 
\end{align*}
$$

$$M_{c}(x,y) = \sum_k w_k^c f_{k}(x,y)$$

- 대충 단점 : 다시 학습해야
- 대충 단점 : GAP(global average pooling) layer가 아니면 CAM을 적용할 수 없다는 한계점이 존재한다.<br>(근데 이건 그냥 GAP layer를 끼워넣으면 되는거라 별 상관은 없다고 생각함)
- A drawback of CAM is that it requires feature maps to directly precede softmax layers, so it is only applicable to a particular kind of CNN architectures performing global average pooling over convolutional maps immediately prior to prediction (i.e. conv feature maps → global average pooling → softmax layer). Such architectures may achieve inferior accuracies compared to general networks on some tasks (e.g. image classification) or may simply be inapplicable to any other tasks (e.g. image captioning or VQA). We introduce a new way of combining feature maps using the gradient signal that does not require any modification in the network architecture. This allows our approach to be applied to off-the-shelf CNN-based architectures, including those for image captioning and visual question answering. For a fully-convolutional architecture, CAM is a special case of Grad-CAM.

## Grad-CAM
- 새로 학습할 필요없이 현재 모델이 이미지의 어느 부분을 주목해 예측했는지 파악할 수 있다.
- output node로부터 last convolution layer로 흐르는 gradient를 추적하여 weight를 얻는다.
- 이때 model에 흐르는 gradients를 다시 활성화 시키기 위해 backpropagation 연산을 수행한다.
- image classification 뿐 아니라, image captioning, visual question answering (VQA) 등 gradients가 흐르는 구조라면 어디에든 적용해볼 수 있다.

(이미지 : gradcam구조)

## 수식
- CAM의 방법보다 Grad-CAM의 방법이 더 효율적이지만 두 방법의 visual explanation이 약간 차이가 있을 수 있지 않을까 의문이 생길 수 있다. 그러나 두 방법이 사실 수식적으로 동일하다는 것은 이미 다음과 같이 증명되었다.

$${a_k}^c = \overbrace{{1 \over Z} \sum_{i}\sum_{j}}^{\text{(1)}} \underbrace{{{\partial y^c}\over{\partial A_{ij}^k}}}_{\text{(2)}}$$
- (1) : global average pooling
- (2) : gradients via backprop

$$L_{Grad-CAM}^c = ReLU\underbrace{(\sum_k a_k^c A^k)}_{(3)}$$
- (3) : linear combination

$$Y^c = \sum_k \underbrace{w_k^c}_{(4)} \overbrace{{1 \over Z}}^{(5)} \sum_i \sum_k \underbrace{A_{ij}^k}_{(6)}$$
- (4) : class feature weights
- (5) : global average pooling
- (6) : feature map

$$F^k = {1 \over Z} \sum_i \sum_j A_{ij}^k$$

$$Y^c = \sum_k w_k^c \cdot F^k$$

$${{\partial}Y^c \over {\partial F^k}} = {{{\partial}Y^c \over \partial A_{ij}^k} \over {{\partial}F^k \over \partial A_{ij}^k}}$$

$${{\partial}Y^c \over {\partial F^k}} = {{\partial}Y^c \over {\partial A_{ij}^k}} \cdot Z$$

$$w_k^c = Z \cdot {{\partial}Y^c \over {\partial A_{ij}^k}}$$

$$\sum_i \sum_j w_k^c = \sum_i \sum_j Z \cdot {{\partial}Y^c \over {\partial A_{ij}^k}}$$

$$Zw_k^c = Z \sum_i \sum_j {{\partial}Y^c \over {\partial A_{ij}^k}}$$

$$w_k^c = \sum_i \sum_j {{\partial}Y^c \over {\partial A_{ij}^k}}$$

$$a_k^c = {1 \over Z}\sum_i \sum_j - {\partial y^c \over \partial A_{ij}^k}$$

## 활용
(이미지 : gradcam1)<br>
(이미지 : gradcam2)
