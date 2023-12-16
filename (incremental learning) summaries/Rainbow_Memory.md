# Rainbow Memory: Continual Learning with a Memory of Diverse Samples

## 2. Related Work
### Blurry-CIL
disjoint-CIL => $\bigcap {T_t} = \varnothing$

blurry-CIL => $\bigcap {T_t} \ne \varnothing$

${T_t}$는 t step에서의 학습되는 classes. 보통 incremental learning setting에서 task마다 class가 전혀 겹치지 않는 것을 가정하지만 이는 연구적인 세팅일 뿐이고, 현실의 상황에서는 task마다 class가 겹칠 수 있다. 
예를 들어 e-commerce 의류 판매량의 데이터를 추가적으로 학습시키는 경우, 계절과 시기별로 majority classes와 minority classes가 차이가 나는 것이지 일부 classes가 아예 존재하지 않는 경우는 아니다.

### online learning and offline learning
CIL에서 현재 task에서 incoming하는 sample에 대해 temporary beffer에 저장시켜 같이 학습하도록 하는 것이 offline learning이며 (unresticted access to a particular task (not to the previous ones),
online learning에서는 sample data가 들어오는대로 한 번(epoch) 통과한 후 버려지며, 일부를 제외하면 memory buffer에 저장시켜 놓지 않기 때문에 여러 epoch에 걸쳐서 학습에 사용할 수 없다.


## 4. Approach

### 4.1 Diversity-Aware Memory Update
blurry-CIL with episodic memory를 효율적으로 다루기 위해 해당 논문은 sample의 다양성을 최대한 향상 시킬 수 있는 memory management 전략을 제안한다.<br>
(we propose a memory management strategy that enhances diversity of samples to cover the distribution of the class by sampling a diverse set of samples which may preserve the boundary of a class distribution)

![image](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/8d9c18b2-3752-45a9-a269-00a79a780225)

sample의 uncertainty를 계산하여 uncertainty가 낮은 순서대로 rehearsal memory를 sampling한다. uncertainty는 perturbed samples의 outputs의 variance를 측정하여 계산된다. 여기서 말하는 perturbattion이란 color jitter, shear, cutout등의 data augmentation을 위한 다양한 transformation 방법들을 가리킨다. 

uncertainty는 perturbed sample $\tilde{x}$가 주어질 때 $p(y=c|x)$에 대한 Monte-Carlo 방식을 통해 계산된다.
$$
\begin{align} p(y=c|x) & = \int_{\tilde{\mathfrak{D}}} p(y=c|\tilde{x}_t)p(\tilde{x}_t|x)d\tilde{x}_t \\
& \approx {1 \over A} \sum\_{t=1}^A p(y=c|\tilde{x}_t)
\end{align}
$$

$x, \tilde{x}, y, A$ : a sample, a perturbed sample, the label of the sample, the number of perturbation methods<br>
$\mathfrak{D}$ :  data distribution defined by the perturbed samples $\tilde{x}$

$$\tilde{x} = f_r (x|\theta_r),~~r=1,...,R$$

$\theta_r$는 $r$-th perturbation의 random factor를 가리키는 hyper parameter이다. 

$$\tilde{x} \sim \sum_{r=1}^R w_r*f_r (x|\theta_r)$$

random variable $w_r, r=\{1,..,R\}$는 categorical binary distribution으로부터 추출되었다. sample의 uncertainty $u(x)$는 다음과 같이 계산된다.

$$S_c = \sum_{t=1}^T \mathbb{1}\_c \arg\max_\hat{c} p(y=\hat{c}|\tilde{x}_t)$$

$$u(x) = 1-{1 \over T}\max_c S_c$$

어떤 sample $x$의 모든 (T개) perturbed sample $\tilde{x}_t$들의 predicted label들 중 가장 빈번한 label이 $\max_c S_c$가 되며 모든 perturbed sample들에 대해 일관되게 같은 label로 예측될 수록 uncertainty가 낮다고 할 수 있다.


![image](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/15cbddec-c80a-4c91-ba6f-d0e4adff5260)

해당 pseudo code는 각 클래스마다 같은 개수의 sample을 rehearsal memory에 $u(x)$가 낮은 순서대로 편성하는 과정을 담고 있다.

### 4.2 Diversity Enhancement by Augmentation

**Mixed-Label Data Augmentation**
task의 iteration이 진행됨에 따라 new task의 sample은 episode memory의 sample과 다른 분포를 따를 가능성이 높다. new task의 classes에 있는 이미지와 memory에 있는 기존 classes의 exemplar를 혼합하기 위해 mixed-labeled DA를 채택하였다. 이러한 mixed-label DA는 class 분포의 변화로 인한 부작용을 완화하고 성능을 향상시킨다.
$$\tilde{x} = \mathbf{m} \odot x_1 + (1-\mathbf{m}) \odot x_2$$

$$\tilde{y} = \lambda y_1 + (1-\lambda)y_2$$

$\mathbf{m}$은 randomly selected pixel region이다.

**Automated Data Augmentation**
mixed-label DA와 더불어 automated DA를 자동하여 cIL 하에서 모델 성능에 여러 DA를 합성하여 augmentation effect를 강화한다. 

## 5. Experimental Setup
blurry-CIL setup을 'BlurryM'으로 설정하는데 여기서 M은 다른 task들로부터 오는 sample들의 portion을 의미한다. 따라서 각 task에는 할당된 major classes의 sample이 (100-M)%로 구성되고 minor classes의 sample이 나머지 M%로 구성된다. 각 task에서 minor classes의 분포는 균형을 유지한다.


(결과는 생략)

