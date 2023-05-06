# On Calibration of Modern Neural Networks (2017)
논문 : https://arxiv.org/pdf/1706.04599.pdf <br>
참고자료 :  https://scikit-learn.org/stable/modules/calibration.html#calibration

Confidence calibration : the problem of predicting probability estimates representative of the true correctness likelihood

## Abstract
- 복잡한 neural network일수록 잘 miscalibration되는 경향이 관찰됨
- 많은 실험을 통해 depth, width, weight decay, batch normalization 등이 calibration에 영향을 미치는 중요한 요인이라는 것을 발견
- 다양한 post-preprocessing calibration methods를 SOTA(2017년 기준) model의 image, text classifcation에 적용하였다.

## Introduction
real word decision making system에서 분류모델은 정확도 뿐 아니라 얼마나 부정확할 수 있는지를 나타내는 것 역시 중요하다. 예를 들어 자율주행 자동차 같은 경우 보행자와 다른 장애물들을 발견해야하는데 detection 모델이 confidently하게 갑작스런 장애물의 존재를 예측하지 못한다면 제동을 위해 다른 센서의 출력에 더 의존해야한다. automated health care 등의 영역 역시 마찬가지이다. 구체적으로 모델은 calibrated confidence measure를 예측과 더불어 제시해야한다. In other words, class label 예측과 관련된 확률은 ground truth correctness를 반영해야한다.

calibrated confidence 추정은 모델의 해석관점에서도 중요하다. classification decision을 해석하기 어려운 경우 사용자와의 신뢰성을 확립하는데 귀중한 추가 정보를 제공하기도 하며, 좋은 probability 추정치를 사용하여 신경망을 다른 확률 모델에 통합할 수도 있다. 예를 들어 음성 인식 분야에서 네트워크 출력을 언어모델과 결합하거나 object detection을 위한 카메라 정보와 결합하여 성능을 향상시킬 수도 있다.


![캡쳐3](https://user-images.githubusercontent.com/59189961/236619537-1982ca13-551c-42c3-88e0-23235f0ddf70.png)<br>
(figure 1)<br>
- CIFAR-100 dataset에 대한 5-layer LeNet과 110-layer ResNet의 분류 결과와 prediction confidence의 분포. 
- ResNet의 정확도가 더 우수하지만 confidence와 match되지 않는다는 것을 확인할 수 있음

**Goal**
- understand why neural networks have become miscalibrated
- identify what methods can alleviate this problem

## Definition


$$
\begin{align*}
& X \in \mathcal{X}\text{ : input} \\
& Y \in \mathcal{Y} = {1,....,K}\text{ : label} \\
& \text{Let h be a neural network with } h(X) = (\hat{Y}, \hat{P}) \text{, where } \hat{Y}\text{ is a class prediction and }\hat{P}\text{ is its associated confidence} \\
\end{align*}
$$

We would like the confidence estimate $\hat{P}$ to be calibrated, which intuitively means that $\hat{P}$ representats a true probability.<br>
(For example, given 100 predictions, each with confidence of 0.8, we expect that 80 should be correctly classified)



$$
\begin{align*}
& \textbf{Perfect Calibration : } \mathbb{P}(\mathcal{Y} = Y|\mathcal{P} = p) = p,  \text{    } \forall{p} \in [0,1]        - (1)\\
\end{align*}
$$

위 정의에서 $\hat{P}$가 continuous random variable이므로 probability는 유한개의 많은 샘플들로 계산될 수 없다. 그렇기 때문에 위 식의 의미를 최대한 포착하는 empirical approximation이 필요하다.

### Reliability Diagrams
= visual representation of model calibration. 

expected sample accuracy를 confidence의 function으로 그리게 된다. finite한 sample로부터 expected accuracy를 추정하기 위해 prediction을 M개의 interval로 그룹화하여 각 bin에서 accuracy를 계산한다.<br><br>

$$
\begin{align*}
& \text{Let }B_m \text{ be the set of indices of samples whose prediction confidence falls into the interval }I_m=({{m-1} \over {M}},{m \over M}].  \\
& acc(B_m) = {1 \over |B_m|} \sum_{i \in B_m} \textbf{1} (\hat y_{i} = y_i) \leftarrow\text{approximation of left-hand of (1)}\\
& conf(B_m) = {1 \over |B_m|} \sum_{i \in B_m} \hat p_i \leftarrow\text{approximation of right-hand of (1)} \\
\end{align*}
$$

<br>그러므로 perfect calibrated model은 모든 $m \in {1,...,M}$에 대해 $acc(B_m) = conf(B_m)$인 상태를 갖는다. 단 reliablity diagram은 주어진 bin에 있는 표본의 비율이 나타나지 않으므로 calibrated된 표본의 수를 추정하는데에는 사용할 수 없다.


### ECE (Expected Calibration Error)
reliablity diagram은 calibration이 얼마나 잘되었는지를 확인할 수 있는 유용한 툴이지만 scalar 값으로 statistic of calibration을 summary하여 표현하는 것 역시 필요하다. miscalibration을 표현하는 한 가지 방법은 confidence와 accuracy의 차이의 기대값을 구하는 것이다.

$$
\begin{align*}
ECE & = \mathbb E_{\hat P} [|\mathbb P (\hat Y = Y | \hat P = P) - p|] \\
& = \sum_{m=1}^M {|B_m| \over n} {|acc(B_m) - conf(B_m)|} \\
\end{align*}
$$

- $n$은 number of samples, acc와 conf의 gap은 calibration gap을 나타낸다. (figure1의 red bars)
- ECE를 primary empirical metric으로 사용하기로 한다.

### MCE (Maximum Calibration Error)
신뢰할 수 있는 confidence 측정이 절대적으로 필요한 고위험 application에서는 conf와 acc간의 worst한 편차를 최소화하는 방향으로 metric을 계산할 수 있다.

$$
\begin{align*}
MCE & = max_{p \in [0,1]} |\mathbb P (\hat Y = Y | \hat P = P) - p| \\
& = \max_{m \in \lbrace1,...,M\rbrace } {|acc(B_m) - conf(B_m)|} \\
\end{align*}
$$

- MCE는 가장 큰 calibration gap이고 ECE는 calibration gap의 가중평균이다.

## Oberserving Miscalibration
![캡쳐2](https://user-images.githubusercontent.com/59189961/236619535-f4d1050e-02e5-471a-8c23-d913fde2a8e6.png)
- Model capacity
  - network의 depth와 width가 증가하면 classification error는 감소하는 반면 model calibration에는 부정적인 영향을 미친다.
- Batch Normalization
  - batch normalization은 별도로 정규화할 필요성을 줄여주고, 어떤 케이스에서는 모델 정확도를 올리기도 한다.
  - 다만 batch normalization과 함께 학습된 모델은 더 miscalibrated하는 경향이 있다. (learning rate같은 hyperparameter와는 관계없)
- Weight decay
  - less weight decay가 calibration에 부정적인 영향을 미치는 사실을 발


![캡쳐1](https://user-images.githubusercontent.com/59189961/236619534-a24ee39c-8831-42d6-8ff7-1ca7fd44fcff.png)
<br>(epoch 250 : when the learning rate is dropped)
- NLL (negative log likelihood)
  - = cross entropy loss in the context of deep learning
  - 간접적으로 model calibration을 측정할 수 있음
  - NLL과 accuracy 사이의 disconnection을 관찰할 수 있었는데 0/1 loss에 overfit되지 않고 NLL에 overfit되기 때문이다.
  - epoch 250 시점 이후 NLL에 overfit되었는데 이게 오히려 classification accuracy에 이로운 결과를 낳았다.
  - $\Rightarrow$ network는 잘 모델링된 probability를 희생하여 더 나은 분류 정확도를 학습함을 확인할 수 있다.
  
## Calibaration Methods
다음 제시되는 방법들은 모두 calibrated probability를 만들어내는 post-processing step이다.<br>
다음 방법들을 적용함에 있어서 hold-out validation set를 필요로 한다.

### For Binary
- Histogram binning
- Isotonic regression
- Bayesian Binning into Quantiles (BBQ)
- Plat scailing

### Extension to Multiclass
- Extension of binning methods
- Matrix and vector scailing

## Result

## Conclusion
