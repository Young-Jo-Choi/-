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

$$$$

$$
\begin{align*}
& \text{Perfect Calibration : } \mathbb{P}(\mathcal{Y} = Y|\mathcal{P} = p) = p, \forall{p} \in [0,1] \\
\end{align*}
$$

ECE, MCE

## Oberserving Miscalibration
- Model capacity
- Batch Normalization
- Weight decay
- NLL

## Calibaration Methods
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
