# TabNet: Attentive Interpretable Tabular Learning (2020)

# Abstract

해당 논문은 tabular데이터에 대해 high performance를 보이고, interpretable한 deep learning 모델을 제안한다. TabNet은 sequential한 attention을 사용해 각 단계마다 어떤 feature들을 고려할지 선택한다. 이러한 방법을 통해 interpretability를 높이고 가장 중요한 feature들에 대해서만 learning capacity를 사용하여 학습의 효율성을 증가시킨다. 

해당 논문에서 여러 데이터셋에 대해 outperform하는 것을 보이며 feature들의 해석가능한 속성과 전체적인 동작에 대한 인사이트를 얻을 수 있음을 입증한다. 마지막으로 tabular data에 대해 unlabeled data가 많을 때 self-supervised learning이 performance를 상승시킬 수 있음을 보인다.

# Introduction

deep learning은 image나 text, audio 등의 분야에서는 좋은 성능을 보여주지만 tabular dataset에서는 아주 높은 성능을 보여주지는 못한다. tabular dataset에서는 decision trees(DTs)의 ensemble 형태의 모델들이 지배적인 위치를 점하고 있다. 이러한 이유로는 다음과 같은 것들을 들 수 있다.

1.  DT-based 접근법은 몇 가지 이점이 존재하는데 
    1. table 형태의 데이터에서 흔히 볼 수 있는 대략적인 hyperplane boundaries의 decision manifold에 대해 효율적이며, 
    2. basic form에서 interpretably가 높고, ensemble 형태에서 많이 사용되는 post-hoc explainability methods가 존재한다. (real world에서의 application시 중요한 부분)
    3. train시에 몹시 빠르다는 이점이 있다.
2. 이전에 제안된 DNN architectures(e.g. CNN이나 MLP의 stack)는 tabular data에 적합하지 않은 방법들이었다. 이런 방법들은 적절한 inductive bias가 부족하여 optimal solutions for tabular decision manifolds를 찾아내는데 실패하고는 한다. 

하지만 그럼에도 불구하고 tabular data에 대한 deep learning은 여전히 가치가 있는데 

1. 아주 큰 데이터셋에 대한 성능 상승을 기대해 볼 수 있으며
2. tree based learning과 다르게 DNNs는 gradient descent에 기반한 end-to-end learning을 통해 다음과 같은 것들을 가능하게 한다.
    1. 여러 type의 데이터(e.g. images)를 tabular data와 함께 학습 가능하게끔 만들며,
    2. DT계열에서의 key aspect인 feature engineering에 대한 필요성을 완화시킬 수 있고,
    3. streaming data에 대한 learning,
    4. data-efficient domain adaptation, generative modeling, semi-supervised learning을 포함한 여러 가지 시나리오에 대한 application을 가능하게 만든다.

해당 논문에서 제시한 DNN architecture인 TabNet의 주요 contribution은 다음과 같다.

1. TabNet은 raw tabular data에 대해 어떠한 전처리 없이 end-to-end로 learning이 가능하도록 만들며,
2. sequential한 여러 decision step을 구성하고 각 step에서의 attention을 통해 어떤 feature를 고려할지 선택한다. 이러한 feature selction은 instance-wise로 이뤄진다.
3. (2)와 같은 방법으로 다른 tabular learning 모델들에 비해 classfication, regression에 대 outperform하는 결과를 보이며, feature의 importance와 어떻게 조합되는지에 대한 것을 시각화하는 local interpretability와 각 featurer가 모델 학습에 어떻게 공헌하는지 정량화하는 global interpretability, 두 가지 관점에서의 interpretability를 얻는다. 
4. masked features를 예측하는 unspuervised pre-training을 통해 성능을 상승시킨다.

# Related Work

생략 (Feature selection, Tree-based learning, Integration of DNNs into DTs, Self-supervised learning)

# TabNet for Tabular Learning

![Untitled](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/d720db4f-5ccc-4295-9177-739c56d69ebd)

conventional DNN에서는 Fig. 3과 같이 DT와 같이 output manifold가 구성되도록 동작한다. 이런 디자인에서 각각의 feature selection은 hyperplane form에서의 decision boundaries를 얻는데 중요한 부분이며 coefficients가 각 feature의 proportion을 결정하는 형태로 각 feature의 linear한 combination에 의해 이뤄진다. 

TabNet은 이러한 기능을 기반으로 하여 조금 더 세심한 설계를 통해 DT보다 뛰어난 성능을 발휘한다.

1. sparse instance-wise feature selection
2. sequential multi-step architecture, where each step contributes to a portion of the decision based on the selected features
3. efficient learning capacity via non-linear processing of the selected features
4. mimicing ensembling via higher dimensions and more steps.

![Untitled 1](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/7c3ad882-f54c-45c6-a165-0a31eaa75dee)

Fig. 4는 TabNet architecture가 tabular 데이터를 어떻게 encoding하는지 보여준다. raw data를 사용하되 global feature embedding을 사용하지 않고 batch normalization(BN)을 사용한다. 

$\text{f} \in \mathbb{R}^{B\times D}$는 $B$ 사이즈 batch의 $D$ dimensional feature로써 각 step의 input으로 들어간다.

${(i-1)}^{th}$ step에서 어느 feature를 사용할지 결정하고 processed된 feature representation이 전체 decision에 통합되도록 만들어낸다. (Fig. 4(a)) 

## Feature selection

- learnable mask $M[i] \in \mathbb{R}^{B\times D}$ for soft selection of the salient features. ($\Sigma_{j=1}^D M[i]_{b,j} = 1$)
- $M[i] = \text{sparsemax}(P[i-1]\cdot h_i(a[i-1]))$는 직전 step의 processed features $a[i-1]$로부터 얻는다.
    - Fig. 4(d)의 attentive transformer
    - $a[i] \in \mathbb{R}^{B \times N_a}$는 Fig. 4(a)의 split에서 얻어진 것임
    - $h_i$ : a trainable function $\in\mathbb{R}^{N_a \times D}$, Fig. 4(d)의 fully connected
- masking은 $M[i] \cdot \text{f}$ 로 진행되며 가장 salient한 feature의 sparse selection을 통해 learning capacity의 낭비를 막는다.
- $P[i]$는 prior scale term으로써 특정 feature가 이전 step들에서 얼마나 사용되어 왔는지를 나타낸다.
    - $P[i] = \Pi_{j=1}^{i} (\gamma-M[j])$, where $\gamma$ is a relaxation parameter
    - $\gamma$가 증가할 수록 더 많은 flexiblity가 여러 decision step에 걸쳐서 주어진다.
    - $P[0]$ is initialized as all ones, $1^{B\times D}$
    - 만일 어떤 feature가 전혀 사용되지 않는다면 corresponding하는 $P[0]$의 entries가 0으로 만들어진다.
- $L_{sparse} = \Sigma_{i=1}^{N_{steps}} \Sigma_{b=1}^B \Sigma_{j=1}^D {{-M_{b,j}[i]log(M_{b,j}[i]+\epsilon)} \over N_{steps} \cdot B}$
    - sparsity를 control하기 위한 loss로써 entropy의 form을 띄고 있다.
    - 즉 masking이 고루 분포하면 엔트로피가 커지지만 한쪽 feature에 몰려있으면 엔트로피가 작아지는 양상을 띈다.
    - overall loss에는 $\lambda_{sparse}$가 곱해져 더해진다.

참고) sparemax 

![sparsemax](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/b0389d47-08b7-40ea-acd6-f5855c26c9f3)

![sparsemax2](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/b24e39ce-3253-456f-81f0-69f7e573f126)

## Feature processing

- Fig. 4(c)의 feature transformer를 이용해 filtered feature를 처리한다.
- 이 과정 중 decision step output과 다음 단계의 information를 위해 분할한다.
    - $[d[i],a[i]] = f_i(M[i]\cdot \text{f})$, where $d[i] \in \mathbb{R}^{B\times N_d}$, $a[i] \in \mathbb{R}^{B \times N_a}$
- 효율적이고 robust한 학습을 위해 feature transformer는 모든 decision step에서 공유하는 layers와 각 decision step에 dependent한 layers로 구성되어있다.
- FC layer-BN layer-residual connection with $\sqrt{0.5}$ 로 이어진 블럭들로 구성되어있음
    - $\text{GLU}(a,b) = a \otimes \sigma (b)$
    - 빠른 학습을 위해 더 큰 batch size를 사용하는데 처음 input features에 적용된 것을 제외하고는 ghost BN이라는 것을 사용해 가상의 batch size $B_V$와 momentum $m_B$를 사용한다.
- decision tree 기반 모델들에서 aggregation하듯이 각 step에서의 output들을 결합해 overall decision embedding을 만든다.
    - $d_{out} = \Sigma_{i=1}^{N_{steps}}\text{ReLU}(d[i])$
- 최종적인 output은 linear mapping을 통해 얻는다.
    - $W_{\text{final}}d_{\text{out}}$

## Interpretability

- TabNet의 feature selection mask는 각 step에서 어떤 feature가 선택되었는지 주목할 수 있게 해준다.
- 만일 $M_{b,j}[i]=0$라면 $i$번째 step에서 $j$번째 feature의 $b$번째 sample은 decision에 아무 기여하지 않았다는 뜻이다.
- 만일 $f_i$가 linear function이라면 coefficient $M_{b,j}[i]$는 $\text{f}_{b,j}$의 importance에 대응한다.
- 각 step의 feature importance를 aggregate하여 정량화하고자 하는데, 각 step의 mask들을 결합해줄 때의 coefficient가 필요하다. 때문에 각 step의 상대적 중요도를 나타낼 수 있는 coefficient를 계산하면 $\eta_{b}[i] = \Sigma_{c=1}^{N_d}\text{ReLU}(d_{b,c}[i])$와 같다.
    - $\eta_{b}[i]$는 $b^{th}$ sample의 aggregate decision contribution at $i^{th}$ step을 나타낸다.
    - 만일 $d_{b,c}[i] <0$이면 relu로 인해 $i^{th}$ step에서는 contribution이 0이라는 의미이다.
- aggregate feature importance mask는 다음과 같다. $M_{agg-b,j} = \Sigma_{i=1}^{N_{steps}}\eta_b[i]M_{b,j}[i] / \Sigma_{j=1}^D \Sigma_{i=1}^{N_{steps}}\eta_b[i]M_{b,j}[i]$

## Tabular self-supervised learning

- Fig. 4(b)의 decoder architecture를 통해 TabNet encoded representation $d_{out}$으로부터 tabular feature를 복원한다.
- missing fature columns을 예측하는 task를 수행한다.
    - binary mask $S \in {\{0,1\}}^{B\times D}$
    - TabNet encoder inputs $(1-S)\cdot \hat{\text{f}}$ and the decoder outputs the reconstructed features $S\cdot \hat{\text{f}}$
- encoder에서 $P[0]=(1-S)$로 초기화하여 모델이 known feature에 주목하도록 한다.
- decoder의 last FC layer는 $S$를 곱하여 unkown feature feature를 만들어낸다.
- reconstruction loss in self-supervised phase :  $\Sigma_{b=1}^B \Sigma_{j=1}^D {|{(\hat{\text{f}}\_{b,j} - \text{f}\_{b,j})\cdot S_{b,j} \over \sqrt{\Sigma_{b=1}^B {(\text{f}\_{b,j} -1/B\Sigma_{b=1}^B \text{f}_b,j)}^2}}|}^2$
    - normalization with the population standard deviation of the ground truth
- $S_{b,j}$는 parameter $p_s$를 지닌 Bernoulli 분포로부터 독립적으로 추출된다.

![image](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/8e3e2260-029c-46fc-b5cc-6cb28ee7d957)

# Exprements

- 각 dataset에서 categorical feature는 learnable한 embedding이 있는 1차원 trainable scalar에 mapping되고, numerical column은 전처리 없이 입력된다.
- appendix에 자세히 나와있지만 TabNet은 hyperparameter에 별로 민감하지 않다.

## Instance-wise feature selection

![Untitled 2](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/eb6726f0-f4c9-4274-b3f7-b55cbe8dcd04)

6개의 synthetic dataset에 대한 비교인데 Syn1\~Syn3는 salient feature가 모든 instance에서 동일하며 salient feature를 global로 선택하면 높은 성능을 얻을 수 있다. Syn4\~Syn6은 salient feature가 instance에 따라 달라지므로 global한 feauture selection은 최적이 아니게 된다. 

(시각화)

![Untitled 3](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/3d7e4f78-fe9d-4954-9cc9-9b661c729b31)

## Performance on real-world datasets

![%EA%B7%B8%EB%A6%BC1](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/ccd46339-1331-4e5b-9f5e-9b48a042cba4)

## Self-supervised learning

![%EA%B7%B8%EB%A6%BC1 1](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/30b30410-a252-4566-b2eb-19a6942f78e5)

# Conclusion

- 해당 논문은 tabular data를 위한 deep learning인 TabNet을 제안하였다.
- TabNet은 sequential한 attention을 사용하여 각 decision step에서 의미있는 subset만을 선택하도록 하였다. Instance-wise한 feature selection을 통해 model capacity가 salient features에 충분히 사용되는 효율적인 learning이 가능하게 하였다. 또한 seleciton mask의 시각화를 통해 해석력을 더욱 높였다.
- TabNet이 다른 모델들에 비해 여러 도메인에 걸쳐 성능이 제일 좋다는 것을 보였으며 unspuervised pre-training을 통해 성능이 상승하는 효과를 보았다.


# (추가)
실제 training 과정에 decoder가 관여하는지

![image](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/180414f5-7576-41f8-9892-218e1dd0af42)
(출처 : pytorch_tabnet 공식 홈페이지 : https://dreamquark-ai.github.io/tabnet/generated_docs/README.html#how-to-use-it)

- 보면 pretrain 이후에 fit하는 것을 확인할 수 있다.
  
![image](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/d30291e1-6cee-443d-afe4-305871db9842)
(출처 : https://github.com/dreamquark-ai/tabnet/tree/develop/pytorch_tabnet)

- pretrain에는 encoder, decoder가 모두 쓰이는데 그냥 모델에는 encoder만 쓰인다. 공식홈페이지의 TabNetClassifier는 TabNet class가 사용된다.
- 즉 pretrain으로 encoder, decoder 모두 학습을 하고 이후 supervised learning에서는 encoder만 학습을 하는 형태이다.
