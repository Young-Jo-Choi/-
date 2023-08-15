# Training data-efficient image transformers & distillation through attention (DeiT)

# Abstract

- 최근(?) convolution layer를 사용하지 않고 순수하게 attetion에 기반해 image를 다루는 neural network가 소개됨 (ViT, vision transformer)
- 높은 performance를 보이는 ViT는 수억 단위의 image들로 사전학습된 것임
- 이 연구에서 외부 데이터셋을 쓰지 않고 오직 imagenet만을 사용해 ViT를 학습시켰고 이 과정에서 transformer에 특화된 teacher-student strategy를 도입하였음

# 1. Introduction

- Convolutional neural network는 classification등 image를 이용한 task에서 main design paradigm을 차지하고 있다.
- Dosovitskiy 등에 의해 소개된 vision transformer(ViT)는 raw image patches를 NLP에서의 token처럼 사용해 transformer achitecture를  가져온 것이다. ViT 논문에서 image classification에 대해 훌륭한 성능을 보였는데 이는 JFT-300M이라는 3억개의 image 데이터셋에 사전학습 시키고 원하는 데이터셋에 fine-tuing하는 방식으로 얻은 것이다. 해당 논문에서도 모델에 대해 “do not generalize well when trained on insufficient amounts of data”라고 결론지었었다.
- 이 연구에서는 Imagenet 하나만을 training set으로 사용했으며, 단일 8-GPU를 사용해 3일 가량의 시간(pretraining : 53H, finetuning : 20H) 을 사용해 비슷한 개수의 파라미터와 효율성을 가진 convnet과 경쟁할 수 있는 수준으로 훈련했다.
- Figure1에서 보이듯 이전 연구보다 더 나은 성능을 기록했으며 ablation study에 상세한 parameter 정보를 기록하고 있다.

![캡쳐30](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/071df246-2aed-4227-b179-ab56708a8d8b)

- distillation 같은 경우 token-based strategy라는 transformer에 최적화된 방법을 소개하고 일반적인 distillation을 능가하는 것까지 같이 보였다.
    - distillation token이라는 것을 사용하는데 ViT의 class token과 동일한 역할을 수행한다. 단지 class token은 true label을 예측하는데에 쓰이고 distillation token은 teacher model이 만들어내는 label을 reproduce하는데에 쓰인다.
- Imagenet으로 훈련시킨 DeiT 모델은 CIFAR-10, CIFAR-100, Oxford-102 flowers, Stanford Cars and iNaturalist-18/19 등의 데이터에 대한 classification 역시 downstream task로써 전이학습 시킬 수 있다.

# 2. Related work

## Image Classification

ViT가 convolution layer 없이 Imagenet에 대해 높은 성능을 보였지만 large volume의 데이터에 pre-training하는 과정이 요구된다. 일반적으로 ViT는 inductive bias가 부족하다고 얘기하며 적은 학습 데이터셋만으로 테스트 데이터셋에 일반화하는 성능은 CNN이 더 낫다고 알려져 있다.

## Knowledge Distillation

Hinton 등에 의해 소개된 학습 방법으로 student 모델에게 teacher 모델이 prediction하는 soft label을 배우게 하는 것이다. 깊은 레이어와 많은 모델 파라미터를 통해 좋은 예측 성능을 가진 teacher 모델의 지식을 비교적 얕은 레이어와 적은 파라미터를 가진 student 모델에게 전달시켜 모델을 경량화하는 방법이다. teacher 모델의 softmax output vector를 student 모델이 배워야하는 정답 값인 것처럼 학습하는 식으로 이루어진다.

# 3. Vision transformer: overview

architecture에 대한 자세한 설명은 생략

## class token

첫 번째 layer 이전에 이미지들의 patch token에 더해지는 trainable vector이며 class를 예측하는데에 필요한 정보를 담고 있다. 이미지를 N개의 token으로 처리한다면 class token이 앞에 더해져 N+1개를 input으로 받게 되며, 이런 architecture는 self-attention을 통해 patch tokens과 class token 사이에서 정보를 분산시키게 된다. 

## Fixing the positional encoding across resolution

“Fixing the train-test resolution discrepancy(2019)” 에서는 ViT를 훈련할 때는 lower resolution을 사용하고 large resolution에서 fine-tuning하는 것이 좋다고 했다. 이런 세팅이 전체 훈련 속도를 높이고 일반적인 데이터 증강 방식에서 정확도를 향상하게 한다. 입력 image의 해상도를 높이면 patch 크기는 동일하게 유지되므로 patch의 수 N은 변경된다. 

# 4. Distillation through attention

강력한 image classifer를 teacher 모델로 갖고 있다고 가정한다. 이 teacher 모델은 convnet일 수도 있고, classifier들의 mixture일 수도 있다. 이 teacher 모델을 이용해 어떻게 transformer를 학습시키는지에 대해 다룬다. 

hard distillation **vs** soft distillation
classical distillation **vs** distillation token

## Soft distillation

teacher 모델의 softmax와 student 모델의 softmax 값 사이의 KL(kullback-Leibler) divergence를 최소화하는 방향으로 학습한다. 

$L_{global} = (1-\lambda)L_{CE}(\phi(Z_s),y) + \lambda{\tau}^2KL(\psi(Z_s/\tau),\psi(Z_t/\tau))$

$\tau$ : the temperature for the distillation
$\lambda$ : the coefficient balancing the KL divergence loss and the cross-entropy on the ground truth labels $y$, and $\psi$ the softmax function.

## Hard-label distillation

해당 논문에서는 teacher 모델의 hard decision을 true label로 취하는 distillation의 variant를 소개한다. $y_t=\text{argmax}_cZ_t(c)$ 를 teacher 모델의 hard decision으로 하고 해당 distillation의 objective function은 다음과 같다. 

$L_{global}^{hardDistill} = {1 \over 2}L_{CE}(\psi(Z_s),y) + {1 \over 2}L_{CE}(\psi(Z_s),y_t)$

주어진 image에 대해 teacher 모델의 hard label은 특정한 data augmentation에 의해 변할 수 있다. 

hard label은 label smoothing을 통해 soft label로 변환시킬 수 있는데, true labeld로 1-$\epsilon$의 확률을 갖고 나머지 class들이 $\epsilon$의 확률을 나눠 갖는 방식이다. 해당 논문에서는 $\epsilon$=0.1로 고정하였다. 

![캡쳐26](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/8cc00be8-dacd-49d2-a881-fe8ffb0ee791)

## Distillation token

- ViT에서 쓰이는 기존의 class token에 더해 distillation token이라는 것을 initial embedding 단계에서 더하게 된다. ([class token, patch tokens, distillation token])
- distillation token은 class token과 비슷하게 동작하는데 self-attention을 통해 다른 embedding과 interact하며 마지막 layer 이후 계산된 값을 내뱉게 된다. target objective는 total loss의 distillation component로 주어지게 된다.
- 이 distillation embedding은 일반적은 distillation과 같은 방식으로 teacher의 output으로부터 우리 모델을 학습하도록 만들며 class embedding과 상호보완적인 관계를 유지한다.
- 실험을 통해 class token과 distillation token간 평균 cosine 유사도는 0.06이었으나 각 layer를 통과해가며 class embedding과 distillation embedding이 점점 유사해지는 현상을 발견했다. 마지막 layer에서의 cosine 유사도는 0.93으로 관찰하였다. 완벽히 1이 되지는 않는데 이는 각각이 비슷하기는 하지만 똑같지는 않은 target을 목표로 하기 때문이다.
- distillation token이 있음으로 모델에 무언가를 더하는 역할을 하게 된다. psuedo-label을 사용하는 대신 같은 target label을 사용해 두 개의 class token으로 학습 시키게 되면 cosine 유사도 0.999의 거의 비슷한 벡터로 수렴하게 된다. 이런 addtional class token은 classification performance에 아무런 것도 가져다 줄 수 없다. 반면 해당 논문에서 제시된 distillation 전략은 vanilla distillation에 비해 상당한 성능 향상을 보여준다.

## Fine-tuning with distillation

더 높은 resolution의 fine tuning 단계에서는 true label과 teacher prediction을 모두 사용한다. true label만을 이용해 테스트 해본 결과 teacher의 이점이 줄어들고 성능이 저하된다.

## Classifier with our approach : joint classifier

테스트 시점에 모델이 만들어낸 class embedding과 distillation embedding은 모두 linear classifer와 연결되어 라벨을 추론할 수 있다. 그러나 해당 논문에서 사용하는 방법은 이 두 개의 head를 late fusion하는 것으로, 두 classifer에 softmax 출력을 추가하여 예측을 수행한다. 

# 5. Experiments

기본적으로 사용하는 모델은 vanilla ViT와 동일하며 convolution 층을 갖지 않는다.  유일한 차이점은  training strategy와 distillation token의 사용 여부이다. 또한 pre-training에서는 MLP hea를 사용하지 않고 linear classifier만을 사용한다. 기본적으로 ViT-B와 DeiT-B라는 prefix를 붙여 모델 결과를 구분하였고, larger resolution에서 DeiT-B를 fine-tune하면 DeiT-B↑384와 같이 resolution을 뒤에 붙였다. 해당 논문에서 사용한 distillation procedure를 사용하면 이를 구분 짓기 위해 DeiT⚗와 같이 표기하였다. 

ViT-B(DeiT-B)의 파라미터는 D=768, h=12, d=D/h=64로 고정하였다. 
(ViT-B : base / ViT-L : large / ViT-H : huge)

![캡쳐24](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/95560211-d048-4446-b4b4-dc26fa286357)

## 5.2 Distillation

![캡쳐25](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/6b27cffb-9c4d-4ac3-abc2-2820a54a8bcd)

Table 2는 서로 다른 teacher architecture를 사용하여 distillation한 결과를 보여준다. convnet이 가장 best teacher인 것을 확인할 수 있는데 이는 inductive bias를 상속받기 때문으로 보인다. 추후의 모든 실험에서 default teacher는 RegNetY-16GF(84M parameters)로 고정한다.

![캡쳐28](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/321913e1-24cb-40e4-a2b0-1e2ab59db975)

### comparison of distillation methods

서로 다른 distillation strategy에 따른 결과는 table3에 나와있는데 hard distillation이 soft distillation에 비해 더 나은 성능을 보여준다. (심지어 class embedding만을 사용할 때 역시 마찬가지이다.) 가장 성능이 좋은 것은 class+distillation embedding을 동시에 사용했을 때인데 이는 section4를 통해 제시된 전략(late fusion)이고 두 개의 token이 분류에 유용한 정보를 상호보완적으로 제공하는 것을 확인할 수 있다. 

distillation token이 미세하게나마 class token보다 더 나은 결과를 주는 것을 볼 수 있는데 이는 convnet의 inductive bias로부터 더 많은 benefit을 가져왔기 때문으로 보인다.

### Agreement with the teacher & inductive bias?

실제로 훈련을 용이하게 하는 inductive bias를 계승하고 있을까?

![캡쳐29](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/9bb4c98d-a990-4bae-b7b1-6e6e05f10064)

table4는 각 모델 간 예측의 불일치한 비율을 나타낸 것인데 논문의 distilled model이 scratch부터 학습한 transformer보다 convnet과 더 연관되어있는 것을 확인할 수 있다. 조금 더 상세하게는 distillation embedding으로부터의 추론과 convent과의 연관성은 class embedding의 추론이 convnet과 연관된 정도보다 더 강하다는 것을 확인할 수 있다. class embdding은 convnet보다 distillation없이 학습한 DeiT와 더 유사하다. class+distillation embedding의 경우 유사도 측면에서 중간 수치인 것을 확인할 수 있다.

## 5.3 Efficiency는 생략

![캡쳐30](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/071df246-2aed-4227-b179-ab56708a8d8b)

## 5.4 Transfer learning: Performance on downstream tasks

Imagenet에 대한 평가를 통해 DeiT가 좋은 성능을 낸다는 것을 확인하였지만 transfer learning을 통한 다른 dataset에 대한 성능 역시 측정하여 DeiT의 generalization을 확인하였다.

![캡쳐33](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/02b489f4-24ad-4e93-a05d-b8003a8e2d58)

# 6. Ablation

![캡쳐32](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/0af7ce8e-0a17-4a24-9e35-452b7f565544)

## hyper-parameter

자세한 설명은 table9로 갈음함

![캡쳐31](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/c36c2e8a-d624-407b-8cdb-8594928615ad)

