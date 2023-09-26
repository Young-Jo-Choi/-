# Scaling & Shifiting Your Features: A New Baseline for Efficient Model Tuning

# Abstract

- pre-trained된 모델에 fine-tuning을 할 때 보통 모든 parameter를 tuning (full fine-tuning)하거나 혹은 마지막 linear layer만 tuning (linear probing) 하는 방식을 따른다. 하지만 full fine-tuning은 효율적이지 못하고, linear probing은 정확도 하락이 발생할 수 있다.
- 이 paper에서는 SSF라고 불리는 parameter efficient한 fine-tuning 방법을 소개한다. 오직 **S**cale and **S**hifting the deep **F**eatures extracted by a pre-trained model하면 되는 것으로 full fine-tuing에 버금가는 performance를 얻을 수 있다.
- 기존의 VPT 등은 transformer 계열에만 사용할 수 있었다면 SSF는 transformer든, CNN이든, MLP든 모두 사용할 수 있으며, train phase에서 추가로 학습한 parameter를 original parameter와 합쳐 inference 효율성을 높일 수 있다.

# 1. Introduction

- 데이터셋의 스케일과 모델 사이즈가 매우 커지면서 pre-trained 모델을 down stream task에 적용해 더 나은 performance와 더 빠른 수렴을 달성하고자 하는 것이 일반적인 경향이 되었다.
- 일반적으로 사용하는 full fine-tuning은 매우 heavy한데 이로 인해 모델이 작은 데이터 세트에 over-fitting되어 fine-tuning 후 다른 새 작업에는 사용할 수 없게 된다. 결과적으로 각 작업마다 전용 parameter를 저장해야 하고 이는 엄청난 양의 저장공간을 차지하게 된다.
- linear probing은 full fine-tuning과 비교해 더 낮은 performance를 만들어낸다.
- 이에 대한 대안으로 VPT(visual prompt tuning)이라는 것이 나왔었는데 learnable한 prompt를 original image token에 더해서 input으로 넣고 self-attention을 통해 다른 image token들과 interact 하도록 하는 방법이다. linear probing에 비해 분명히 나은 성능을 보이지만 두가지 문제점이 존재한다.
    - VPT는 다양한 downstream task에 대해 prompt의 수를 조정하여 task-dependent한 learnable parameter space를 도입한다. prompt의 수는 task에 따라 민감하게 결정된다. (ex) Clevr/count에서는 200개의 prompt, Flowers102에서는 1개의 prompt)
    - VPT 및 다른 adapter 기반의 방법은 원래의 pre-trained 모델에 비해 추론 단계에서 추가적인 parameter와 computationalk cost가 발생한다. 이러한 방법은 backbone architecture을 변경하므로 이미 edge device에 배포된 모델의 경우 구조가 자주 수정되고 workload가 과중해질 수 있다.
- 이러한 문제를 해결하기 위해 feature에 대해 Scale과 Shifting만을 수행하는 parameter-efficient fine-tuning 방법을 고안해냈다. 아이디어는 upstream task와 down stream task의 데이터 distribution이 다르다는 것에서 기인했다.

# 2. Related Work

## 2.1 Model Families

CNN-based, transformer-based, MLP-based 

이하 생략

## 2.2 Pre-training and Fine-tuning

생략

## 2.3 Feature Modulation

Batch normalization, Layer normalization, Group normalization

이하 생략

## 2.4 Model Re-parameterization

model re-parameterization은 inference의 효율성을 높이는데 많이 쓰이는 방법이다. batch normalization 같은 경우 학습된 batch normalization layer의 parameter가 그 앞에 쌓인 convolution layer에 병합된다.  이러한 tehcnique은 네트워크의 여러 branch를 하나의 branch로 병합하는 데에도 사용된다. (논문의 SSF 방법도 inference의 계산 비용을 낮추기 위해 해당 technique이 사용됨)

# 3. Approach

## 3.1 Preliminaries

### Transformers

생략

### Adapter

효율적인 tuning을 위해 transformer layer에 삽입된다. 적은 trainable parameter를 가지는 일종의 bottleneck module로써 feature를 down-projection 한 뒤 non-linear activation을 통과시킨 뒤 원래의 dimension으로 up-projection 시킨다.

given the input $x \in \mathbb{R}^{{(N^2 + 1)}\times d}$, 
the output is calculated by  $out=[W^{up}\phi(W^{down}x^T)]^T$,
where $W^{down} \in \mathbb{R}^{d^{\prime} \times d}$ ,    $(d^{\prime} \ll d)$,     
$W^{up} \in \mathbb{R}^{d \times d^{\prime}}$

### VPT

learnable parameter(prompt)를 embedding 이후의 input space에 넣는 방식으로 original image token들과 self-attention을 수행하도록 한다. fine-tuning 하는 동안 backbone의 weight는 freeze되고 오직 prompt의 parameter들만 업데이트된다.

assuming that the input is $x \in \mathbb{R}^{{(N^2 + 1)}\times d}$, denoted the inserted promprts as $p \in \mathbb{R}^{n \times d}$, where the n is the number of prompts, the combined tokens $x^{\prime} = [x;p] \in \mathbb{R}^{(N^2 + n + 1)\times d}$

## 3.2 Scaling and Shifting Your Features for Fine-tuning

해당 연구는 pre-trained model의 deep한 feature를 modulate하는 scale과 shift에 대한 factor를 소개하고 이를 통해 target dataset의 distribution을 맞춘다. 해당 방법을 통해 다음 다섯 가지 주요 성질들을 만족시킨다.

- SSF는 full fine-tuning과 동등한 성능을 낸다.
- 모든 downstream task는 다른 task에 의존하지 않고 독립적으로 모델에 입력할 수 있다.
- 모델은 매주 적은 parameter만을 학습한다.
- 각 task마다 prompt 수를 조정하는 VPT와 다르게 SSF에서는  task가 변경되어도 fine-tuning을 위한 parameter set는 변하지 않으므로 multi-task learning 혹은 continual learning을 위해 더 많은 task를 추가하여 parameter를 추가로 fine-tuning하는 것이 가능하다.
- scaling과 shifting이 linear transformation 과정이기 때문에 inference phase에서 불필요한 parameter나 계산 비용을 줄일 수 있다.

### Design of SSF

![num1](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/734ea1d1-3d9a-49a6-b093-52bc05c7d2c2)

upstream task에서 pre-trained된 모델에서, 각 OP(operation) 다음에 SSF-ADA를 넣어 scale&shift 연산을 수행한다. OP들은 multi-head self-attention(MSA), MLP, layer normalization(LN) 등을 포함한다.  fine-tuning 동안, pre-trained weight은 frozen되고 SSF-ADA의 parameters만 update된다.  (figure2 참고, (b)와 (c)는 분리된 그림임)

### Re-parameterization

$y=\gamma \odot x + \beta = \gamma \odot(w*t+b)+\beta = (\gamma\odot w)*t + \gamma\odot b+\beta$

where $w,b$ are the weight and bias term, respectively. * represents the ‘convolution’ operation in the convolutional layer or the ‘multiplication’ operation in the MLP layer.

given the input $x\in \mathbb{R}^{(N^2 + 1)\times d}$, the output $y \in \mathbb{R}^{(N^2+1) \times d}$ is calculated by $y = \gamma \odot x + \beta$

where, $\gamma \in \mathbb{R}^d$ and $\beta \in \mathbb{R}^d$ are the scale and shift factors, respectively.

# 4. Experiments

![num2](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/3d481a0c-b603-40c7-92ec-4fc3bdcbe97b)

설명은 생략 (figure5에 다 나와있음)

![image](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/6c1c5843-6776-443c-a437-a5164740258a)

시간 자체는 큰 차이가 나지 않음

## 4.3 The Impacts of Different Designs

table 6은 pre-trained된 ViT-B/16 model을 CIFAR-100에 tuning시킨 것이다.

![num3](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/40d6b53f-ea4e-42b9-8277-45243422e1ea)

(a) : 몇 개의 layer에 SSF-ADA 모듈을 붙여 tuning 했을 때 성능이 차이 나는지에 대한 결과

(b) : 모든 OP에 SSF-ADA를 넣은 것을 기준으로 어느 OP에서 tuning을 제외시켰을 때 성능차이가 나는지에 대한 결과

(c) : SSF-ADA에 대해 초기화를 어떻게 시켰는가에 대한 결

(d) : SSF-ADA의 각 component 하나씩을 제외했을 때 성능차이가 나는지에 대한 결과, only norm은 Layer normalization에만 Scale and shift factor를 fine-tune했을 때이고 scalar scale은 scale factor를 행렬이 아닌 scalar 값으로만 조정한 경우 

## 4.5 Visualization and Analysis

![num4](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/4eeaa2eb-517e-4047-87c7-a29833005e47)

figure5는 full fine-tuning에 비해 각 layer의 output feature들이 얼마나 유사한지 나타낸 것임, 마지막 layer에서 SSF는 full fine-tuning과 가장 유사한 feature들 가지며, 정확도 역시 제일 근접하다.
