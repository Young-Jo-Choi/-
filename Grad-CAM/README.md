# Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
논문 : https://arxiv.org/pdf/1610.02391.pdf <br>
Abstract We propose a technique for producing ‘visual explanations’ for decisions from a large class of Convolutional Neural Network (CNN)-based models, making them more transparent and explainable.

![gradcam1](https://user-images.githubusercontent.com/59189961/227709603-27a562fe-d456-4610-bb07-61309de2d129.jpg)

- Weakly-supervised localization : 학습할 이미지에 대한 정보보다 예측해야할 정보가 더 디테일한 경우를 말함

## CNN
![CNN구조](https://user-images.githubusercontent.com/59189961/227709601-486e22c3-d060-461b-8226-745fd57a73ab.JPG)

- 일반적인 Neural Network(Fully Connected Layer, Dense layer)는 각 데이터가 여러 개의 feature로 표현되는 2차원 tabluar 형태의 데이터에 적합하다. (각 input node가 하나의 feature에 대응됨)
- image, text 등의 비정형 데이터의 경우 일반적인 neural network에 input 데이터로 사용할 수 없다. 억지로 각 이미지를 1차원이 되도록 펼치더라도 위치 정보가 모두 깨지게 되므로 제대로 된 분류를 해내기 어렵다.
- 이를 위해 Convolution layer를 통해 image의 특징을 추출한 뒤 이미지를 1차원으로 vector로 재구성(Embedding, Representation)하고 이를 fully connected layer에 넣어 분류를 수행한다.
 

## CAM: Class Activation Maaping
논문 : Learning Deep Features for Discriminative Localization.<br>
- 일반 CNN architecture의 output만으로는 분류 결과에 대한 모델이 어디를 주목하는지 알 수가 없다.
- 보통의 deep learning은 내부 parameter를 일일히 확인하기 어려운 black box 구조이기 때문에 모델 해석의 관점에서 어려움이 따른다.

![feature_maps](https://user-images.githubusercontent.com/59189961/227709602-e6bc05cd-db2e-4bc7-8350-83dcee5852ac.jpg)

<--- shallower layer's activation maps 　 　　|　　　  deeper layer's activation maps ---><br>
- layer가 얕을 수록 low-level feature를 읽는다.
- layer가 깊어질 수록 조금 더 고차원적인 특징(class specific하도록 semantic한 정보)을 포착한다.
- 때문에 last layer의 activation map에 주목하여 model이 실제 주목하는 영역을 살펴볼 수 있다.

![CAM](https://user-images.githubusercontent.com/59189961/227709594-6b5519a9-1e2d-4068-9c07-e9598487f76d.jpg)

- 하나의 이미지로부터 얻어지는 last convolution layer의 activation map은 여러 channel이 쌓여있는 3D array이다. (channel 별로 2D image) 이때 channel별로 가중치(weights)를 구해서 각각의 2D image를 가중합하여 얻어지는 image를 각 input image에서 얻어지는 최종 결과물로 하겠다는 것이 아이디어.
- last convolution layer의 output map을 global average pooling하여 channel별로 하나의 값을 얻는다. (gloal average pooling은 2D image 전체를 단순 평균한다는 뜻) 이때 channel 개수만큼으로 얻어진 vector로 output node를 예측할 수 있도록 finetuning을 하고 각 node를 잇는 weight값들이 계산된다.
- predicted된 node(class)로 연결된 weight들을 각각 channel의 가중치로 취급하여 모든 channel의 이미지를 가중합한 것을 해당 class에 대한 CAM으로 하여 visualization한다.

$$ 
\begin{align*}
S_c &= \sum_k w_k^c F_k \\
&= \sum_k w_k^c \sum_{x,y} f_{k}(x,y) \\
&= \sum_{x,y}\sum_k w_k^c f_{k}(x,y) 
\end{align*}
$$

$$
\begin{align*}
& f_{k}(x,y) \text{ represent the activation of unit k in the last convolutional layer at spatial location } (x,y) \\
& F_k \text { indicates the global average pooling of } f_{k}(x,y) \\
& w_k^c \text{ indicates the importance of } F_k \text{ for class } c \\
& S_c \text{ indicates the logit for the class } c, \text{ and probability of class c is given by } {exp(S_c) \over \sum_c exp(S_c)}
\end{align*}
$$ 

$$M_{c} \text{ : class activation map for class c, where each spatial element is given by}$$

$$M_{c}(x,y) = \sum_k w_k^c f_{k}(x,y)$$

- 단점
  - 모델의 fully connected layer단의 architecture를 수정해야한다.
  - finetuning 과정을 다시 거쳐야한다.


## Grad-CAM
- We introduce a new way of combining feature maps using the gradient signal that does not require any modification in the network architecture.
- 새로 학습할 필요없이 현재 모델이 이미지의 어느 부분을 주목해 예측했는지 파악할 수 있다.
- output node로부터 last convolution layer로 흐르는 gradient를 추적하여 weight를 얻는다.
- 이때 model에 흐르는 gradients를 다시 활성화 시키기 위해 backpropagation 연산을 수행한다.
- image classification 뿐 아니라, image captioning, visual question answering (VQA) 등 CNN 기반의 gradients가 흐르는 구조라면 어디에든 적용해볼 수 있다.

![gradcam구조](https://user-images.githubusercontent.com/59189961/227709607-c4e8ba68-ac5d-4a7f-8e34-a739d1b8ed52.jpg)

- 예측된 class를 c라고 할 때 class c로 예측하기 위한 score값을 $y_c$라고 한다. 이 y_c를 원하는 feature map(보통은 마지막 convolution layer)까지 backpropagation을 한다.
- backprop된 gradient는 feature map과 동일한 size로 생성된다. (feature map의 채널별 모든 pixel이 output 예측에 어떤 영향을 미쳤는지 나타내는 것이기 때문)
- 이 gradient를 channel-wise하게 평균한다. 그러면 channel 개수와 동일한 output이 나오고 이를 가중치 삼아 feature map을 channel별로 가중평균한다.


$${a_k}^c = \overbrace{{1 \over Z} \sum_{i}\sum_{j}}^{\text{(1)}} \underbrace{{{\partial y^c}\over{\partial A_{ij}^k}}}_{\text{(2)}}$$

$$
\begin{align*}
& A^k \text{ k-th channel of feature map } a_k^c \\
& a_k^c \text { : } A_k \text {가 전체적인 feature map에서 c를 예측하는데 어느정도 영향을 미쳤는가에 대한 가중치} \\
& \text{(1) : global average pooling} \\
& \text{(2) : gradients via backprop : } A_k \text{의 각 pixel들이 c를 예측하는데 있어 얼마나 유효한가에 대한 값} \\
\end{align*}
$$

$$$$

$$$$

$$L_{Grad-CAM}^c = ReLU\underbrace{(\sum_k a_k^c A^k)}_{(3)}$$

$$
\begin{align*}
& \text (3) : linear combination\\
& \text{results in a Grad-CAM image (=coarse heatmap) which has same size as the convolutional feature maps} \\
\end{align*}
$$

- class 예측에 긍정적인 영향을 미치는 픽셀에만 관심이 있기 때문에 linear combination 이후 ReLU를 적용한다. 음수 픽셀은 다른 class에 속할 가능성이 높으며 ReLU를 하지 ㅇ낳을 경우 원하는 class 외의 것을 highlight하여 더 나쁜 성향을 보이는 것을 확인했다.



$y^c$는 꼭 image classification의 logit일 필요는 없다. image captioning이나 QA 같은 미분가능한 action이 포함된 어떤 task라도 가능하다.




###  Grad-CAM generalizes CAM

- CAM의 방법보다 Grad-CAM의 방법이 더 효율적이지만 두 방법의 visual explanation이 약간 차이가 있을 수 있지 않을까 의문이 생길 수 있다. 그러나 두 방법이 사실 수식적으로 동일하다는 것은 이미 다음과 같이 증명되어있다.


$\bullet \mathbf{ first}$

$$
\begin{align*}
& Y^c = \sum_k \underbrace{w_k^c}_{(4)} \overbrace{{1 \over Z}}^{(5)} \sum_i \sum_k \underbrace{A_{ij}^k}_{(6)}\\
& \text{(4) : class feature weights}\\
& \text{(5) : global average pooling}\\
& \text{(6) : feature map}\\
\end{align*}
$$

$$
\begin{align*}
& \rightarrow Y^c = \sum_k w_k^c \cdot F^k \\
& (F^k = {1 \over Z} \sum_i \sum_j A_{ij}^k) \\
\end{align*}$$

$\bullet \mathbf{ second}$

$$
\begin{align*}
& {{\partial}Y^c \over {\partial F^k}} = {{{\partial}Y^c \over \partial A_{ij}^k} \over {{\partial}F^k \over \partial A_{ij}^k}} \\
& \rightarrow {{\partial}F^k \over \partial A_{ij}^k} = {1 \over Z} \\
& \rightarrow {{\partial}Y^c \over {\partial F^k}} = {{\partial}Y^c \over {\partial A_{ij}^k}} \cdot Z \\
\end{align*}$$

$$
\begin{align*}
& {{\partial}Y^c \over {\partial F^k}} = w_k^c \\
& \rightarrow w_k^c = Z \cdot {{\partial}Y^c \over {\partial A_{ij}^k}} \\
\end{align*}
$$

$\bullet \mathbf{ third}$

$$
\begin{align*}
& \sum_i \sum_j w_k^c = \sum_i \sum_j Z \cdot {{\partial}Y^c \over {\partial A_{ij}^k}} \\
& \rightarrow Zw_k^c = Z \sum_i \sum_j {{\partial}Y^c \over {\partial A_{ij}^k}} \\
& \Rightarrow w_k^c = \sum_i \sum_j {{\partial}Y^c \over {\partial A_{ij}^k}} \\
& \text{Up to a proportionality constant } {1 over Z} \text{ that gets normalized out during visualization,} \\
& \text{the expression for } w_k^c \text{ is identical to } a_k^c \text{used by Crad-CAM} \\
\end{align*}$$

### Guided Grad-CAM
![guided_gradcam](https://user-images.githubusercontent.com/59189961/227724134-4e5aabbb-5a4d-4eaf-b033-81136a0f3ac3.jpg)

- Guided BackPropagation은 ReLU 계층을 통해 음의 gradients가 억제되는 이미지와 관련해 gradient를 시각화한다. 즉 neuron이 억제하는 pixel이 아닌 neuron에 의해 발견되는 pixel을 잡아내는 것을 목표로 한다. (좌측 열) --> high-resolution이 목표
- Grad-CAM이 class마다의 image의 영역을 나타내기는 하지만 왜 해당 클래스로 예측했는지와 같은 세부 사항을 highlight하는 능력은 부족하다. 즉 가운데 열의 이미지를 보면 highlight되는 영역이 확인은 가능하지만 왜 'cat'과 'dog'가 되는지에 대한 정보는 불확실하다. (가운데 열) --> class-discriminative가 목표
- 때문에 Guided Backpropagation과 Grad-CAM을 element-wise muliplication하여 섞어 두가지의 best한 측면을 결합한다. (우측 열) <br>이때 element-wise하게 결합하여는 두 결과의 size가 맞지 않을 수 있는데 Grad-CAM의 결과인 $L_{Grad-CAM}^c$를 bilinear interpolation하여 upsampling하여 사이즈를 맞춘다. <br>(구조는 최상단 그림 참조)


## Diagnosing image classification CNNs with Grad-CAM
### Analyzing failure modes for VGG-16
(이미지 : failure)

- 예측에 실해한 이미지만 보고 사람의 눈으로 원인분석하기는 쉽지 않지만 Guided Grad-CAM으로 visualize함으로써 모델이 이미지를 어떻게 바라봤는지 확인할 수 있다.

### Effect of adversarial noise on VGG-16
(이미지 : adversarial)

- 이미지에 없는 Airliner에는 높은 확률(>0.9999)을 할당하고 실제 존재하는 dog와 cat에는 낮은 확률을 할당하도록 pretrained된 VGG-16 모델에 대해 Grad-CAM을 visualize한 결과가 그림과 같다.
- 모델이 cat과 dog가 존재하지 않는다는 것을 거의 확신하고 있음에도 Grad-CAM은 이 범주를 거의 정확하게 포착해낼 수 있다. 이는 Grad-CAM이 adversarial noise에 강하다는 것을 보여준다.

### Identifying bias in dataset

![gradcam2](https://user-images.githubusercontent.com/59189961/227709605-c6b583c5-832e-4de6-b908-95a5641553c2.jpg)
- 간호사 이미지와 의사 이미지에 대해 두번째 열에서 모델이 둘다 간호사로 분류하였다. 간호사는 올바른 분류를 하였지만 의사의 경우 틀린 예측을 하였다. <br>(여기서의 모델은 VGG-16을 기반으로 의사와 간호사만 구분하도록 binary classification task로 finetune된 모델이다.)
- 오분류에 대한 원인을 분석하고자 Grad-CAM을 이미지를 찍었고 모델이 머리카락을 주로 봤다는 것을 확인하였다. 즉 여자인 것을 모델이 확인하고 실제 의사임에도 간호사로 예측하였다.
- 학습시킨 데이터셋의 구성을 살펴보니 의사의 경우 78%가 남자의사로 되어있고, 간호사의 경우 93%가 여자로 되어있었다.
- 이에 남자 간호사와 여자 의사 이미지를 추가하여 모델을 재학습시켰고 모델이 예측을 올바로 수행하였다. 
- unbiased model의 Grad-CAM 결과를 확인해보면 간호사의 경우 상의의 짧은 소매, 의사의 경우 청진기와 상의의 긴 소매에 주목하여 예측하는 것을 확인하였다.

## Conclusion
