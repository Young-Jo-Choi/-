# Multi-Time Attention Networks for Irregularly Sampled Time Series

# Abstract

time series의 데이터를 수집할 때 irregular하게 sampling된 경우가 매우 많고 이는 일반적인 딥러닝 모델에서 중요한 challenge 중에 하나이다. 해당 논문에서는 EHR 데이터의 여러 상황에서 motivation을 얻었으며 주로 sparse, irregularly sampled, multivariate한 데이터를 다룬다.

해당 논문에서는 Multi-Time Attention Networks(mTAN)라는 딥러닝 framework를 개발하였으며 해당 framework는 연속적인 시간값들의 embedding을 학습하고 attention mechanism을 통해 고정된 길이의 representation을 만들어내도록 한다. 다양한 데이터셋에 interpolation과 classification에 대한 성능을 조사하였고 다른 방법들보다 빠르면서도 우수한 성능을 낸다는 것을 보였다.

# 1. Introduction

healthcare, climate science, ecology, astronomy, biology 등 다양한 도메인에서 불규칙(irregular)하게 샘플링된 time series 데이터는 자주 발생하며 machine learning 모델에서 fully-observed하고 fixed size인 feature로 재구성하는 것은 중요한 challenge 중에 하나이다. RNN계열의 모델은 시계열 데이터를 다룰 수 있지만 fully observed된 상태의 벡터를 input으로 받는다.

해당 연구에서는 multivariate하면서 sparse하고 irregularly sampled된 time series데이터를 처리하는 Multi-Time Attention networks (mTAN)를 제안하였다. mTAN은 기본적으로  연속된 시간과 interpolation을 기반으로 하는 모델이다.  주요한 innovation으로는 학습된 continuous-time embedding mechanism과 time attention mechanism을 도입해 이전 방법들에서 fixed similarity kernel을 도입해 representation을 만드는 것을 대체하였다. 이러한 부분들은 mTAN이 이전의 interpolation-based 방법들보다 표현상 flexibility를 더 잘 갖도록 만들어준다. 

해당 연구에서 불규칙하게 sampling된 time series 데이터를 고정된 reference point로 재구성하는데 referece time point를 query로, observed time points를 key로 사용하는 attention mechanism을 사용한다. 또한 encoder-decoder 구조를 사용하여 interpolation과 classification을 처리하는 과정을 end-to-end로 구성하였다. encoder에서는 irregularly하게 sampling된 time series를 받아 reference time points를 이용하여 고정된 길이의 latent representation을 구성하며 decoder는 latent representation을 이용해 observed time points에 대해 reconstruction하는 과정을 거친다. learning은 variational autoencoders (VAE)에 의해 확립된 방식을 사용한다.

해당 논문이 제안하는 contribution은 다음과 같다.

- It provides a flexible approach to modeling multivariate, sparse and irregularly sampled time series data (including irregularly sampled time series of partially observed vectors) by leveraging a time attention mechanism to learn temporal similarity from data instead of using fixed kernels.
- It uses a temporally distributed latent representation to better capture local structure in time series data.
- It provides interpolation and classification performance that is as good as current state-of-the-art methods or better, while providing significantly reduced training times.

# 2. Related work

생략

# 3. The multi-time attention module

- mTAN module : re-represent a sparse and irregularly sampled time series in a fixed-dimensional space, 여러 개의 연속적인 time-embedding과 attention-based interpolation을 사용한다.

**Notation**

- $D = \{(s_n, y_n)|n=1,...,N\}$는 시계열 데이터를 나타내는데 전체 cases의 수 (의료데이터의 경우 환자수)를 의미한다. $s_i$는 하나의 case에 대응하며 $D$개의 dimension를 갖고 $y_i$는 single target value이다. 여기서 dimension는 논문 내에서 feature와 비슷한 의미로 쓰인다.
- case에 따라 또 dimension $d$에 따라 다른 시간대에서 observation이 보이기도 하며 관측된 개수 역시 다를 수 있다. n번째 case가 d번째 feature에 관측값으로 갖고 있는 것들의 길이를 $L_{dn}$으로 나타낸다.
- time series $d$ for data case $n$은 튜플 $s_{dn}=(t_{dn},x_{dn})$으로 나타내며 $t_{dn} = \[t_{1dn},...t_{L_{dn}dn}]$, $x_{dn} = \[x_{1dn},...x_{L_{dn}dn}]$이다. 여기서 $t$는 시간, $x$는 그 시간대에 대응하는 데이터라고 생각하면 된다.

**Time Embedding** 
- time attention module은 contiuous time points를 vector space로 embedding 시키는 것에 기반한다.
- $\phi_h(t)$ 함수의 $h=1,...,H$ 경우의 embedding을 만들어내며 각각의 output size는 $d_r$이다. 이 함수의 embedding $h$의 dimension $i$는 다음과 같다.

$$\phi_h(t)[i]  = \begin{cases} 
		\omega_{0h} \cdot t+\alpha_{0h} & if ~~~~ i=0 \\ 
         	sin(\omega\_{ih} \cdot t +\alpha_{ih}) & if ~~~~ i>0
     \end{cases}$$


- $\omega\_{ih}$와 $\alpha_{ih}$는 learnable parameters이다.

**Multi-Time Attention**
- 위의 time embdding은 연속적인 time point를 받아 $H$개의 다른 $d_r$ dimension space로 embedding한다. 이 section에서는 어떻게 time embedding을 이용해 irregularly sampled되고 sparse한 time series 데이터를 처리하는지 다룬다.
- multi-time attention embeding module인 mTAN($t,s$)은 query time $t$를 받고 D-dimensional한 sparse, irregularly sampled time series 데이터 $s$를 key로 받아 $J$-dimensional embedding을 return한다.

$$
\begin{align} 
	\text{mTAN}(t,\mathbf{s})[j] & = \sum_{h=1}^H \sum_{d=1}^D \hat{x}\_{hd}(t,\mathbf{s})\cdot U_{hdj}\\ 
    	\hat{x}\_{hd}(t,\mathbf{s}) & =\sum_{i=1}^{L_d} \kappa_h (t,t_{id})x_{id}\\ 
        \kappa_h (t,t_{id}) & = {{\text{exp}(\phi_h(t)\mathbf{wv}^T \phi_h(t_{id})^T /\sqrt{d_k}} \over {\Sigma_{i'=1}^{L_d} \text{exp}(\phi_h(t)\mathbf{wv}^T \phi_h(t_{i'd})^T /\sqrt{d_k}}} 
\end{align}
$$

- $w,v$는 learnable
- $\kappa_h (t,t_{id})$는 feature $d$가 갖고 있는 관측값들 중 $i$의 가중치를 나타낸 것인데 $H$개의 time embdding 함수 중 $h$의 관점에서 바라본 것이라는 의미를 가지며 처음 input $t$를 query 시점으로 사용한 것이다.
- $\hat{x}\_{hd}(t,\mathbf{s})$는 실제 관측값들을 가중치에 따라 가중합한 것이다. $x$는 input $s$의 데이터이다.
- 모든 time embedding 함수 $H$개에 대해, 모든 feature $D$에 대해 가중합한 것을 최종 output으로 하며 shape은 $\mathbb{R}^{J}$이다.
- $U_{hdj}$는 learnable한 linear combination weights이다. 해당 weight는 총 $H \times D \times J$개이다.
- 즉 $t$ 시점 기준으로 특정 case(=instance, 사람)에 대한 time series 데이터를 재구성한 것이라는 의미를 갖는다.

**Discretization**
- mTAN module은 $t,s$에 대해 연속적인 함수를 정의하므로 fixed-dimensional vector나 discrete sequence를 input form으로 하는 neural network에 직접적으로 넣을 수는 없다. 하지만 고정된 길이의 reference time points $r = \[r_1,...,r_K]$에 대한 mTAN module의 representation을 사용한다면 이런 문제에 적응할 수 있다.
- auxiliary function $\rho (s)$를 정의해 reference time points를 만들어내도록 한다. 이때 reference set은 $s$의 어느 dimension이라도 모든 observation이 다 채워져있는 time point들로 한다.
- define the dicretized mTAN module mTAND($r,s$) as mTAND($r,s$)[$i$] = mTAN($r_i,s$). 해당 함수는 input으로 reference time points $r$과 time series $s$를 받아 output으로 mTAN embedding의 length $|r|$인 sequence를 만들어낸다. ($\in \mathbb{R}^{J \times K}$)
  
![캡쳐3](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/bbb8c51a-ce8d-49bd-aafa-a8f3fb647576)

- mTAND module의 전체적인 과정은 fig 1에 나타나 있다. $x_1$과 $x_2$는 $x$의 두 변수이고 각 변수에서 만들어진 embedding seqeunce 중 같은 reference time point(=i)에서 만들어진 embedding($\in \mathbb{R}^{J}$)은 linear 층을 통해 $h_i$로 합쳐진다. 즉 여러 변수에 대해 $|r|$의 길이를 갖는 $h=\[h_1,...h_K\]$로 재구성된다.

  
# 4. Encoder-Decoder framework

![캡쳐4](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/a274e805-90bf-4932-84cc-06f51ceae445)
mTAN network 전체는 encoder-decoder 구조로 되어있고 encoder에서는 classification과 중간의 잠재변수 $z$ 생성, decoder에서는 latent states $z$로부터 time series를 interpolation하는 역할을 한다.

**Decoder**
- $z=\[z_1,...,z_K\]$는 $K$ reference time points에서 만들어진 latent states이고, 이 latent states로부터 RNN, mTAN 연산을 통해 모든 dimension의 데이터가 채워져있는 time series $\hat{s}=(t,x)$를 만들어낸다. (자세한 과정은 fig 2와 아래 수식 참고)
- $f^{dec}$는 fully connected 층으로써 맞는 time point와 dimension에 대한 값을 산출하고 이를 평균으로, 고정된 $\sigma^2$를 분산으로 하는 정규분포로 interpolation될 값을 추정한다.

$$
\begin{align} z_k &\sim p(z_k) \\
h_{RNN}^{dec}&= \text{RNN}^{dec}(z)\\
h_{TAN}^{dec}&= \text{mTAND}^{dec}(t,h_{RNN}^{dec}) \\
x_{id}&\sim N(x_{id};f^{dec}(h_{i,TAN}^{dec})\[d], \sigma^2 I \end{align}
$$

**Encoder**
- mTAN과 RNN 연산을 통해 latent state $z=\[z_1,...,z_K\]$를 만들어낸다. (자세한 과정은 fig 2와 아래 수식 참고)
- 만들어진 latent state는 분류에도 사용된다.
- $f\_{\mu}^{dec}$와 $f\_{\sigma}^{dec}$는 decoder와 비슷한 역할로 사용됨 
- $\gamma$는 encoder components의 모든 parameter를 나타낸다.

$$
\begin{align} 
h_{TAN}^{enc} &= \text{mTAND}^{enc}(r,s)\\
h_{RNN}^{enc} &=\text{RNN}^{enc}(h_{TAN}^{enc}) \\
z_k \sim q_\gamma(z_k|\mu_k,{\sigma}^2_k), ~~~ \mu_k &= f_{\mu}^{enc}(h_{k,RNN}^enc), ~~~  {\sigma}^2_k =\text{exp}(f_\sigma^{enc}(h_{k,RNN}^{enc}))
\end{align} 
$$

**Unsupervised Learning**

- we follow a slightly modified VAE training approach and maximize a normalized variational lower bound on the log marginal likelihood based on the evidence lower bound or ELBO. (VAE를 잘 모르는 관계로 해당 부분은 패스)
- $\text{log}p_\theta(x_n|z,t_n)$는 어떤 case $n$에 대해 latent state $z$로부터 모든 feature의 모든 실제 관측된 값들을 가질 log likelihood function을 더한 것
- $D_{KL}(q_\gamma(z|r,s_n)||p(z))$은 모든 reference time points에 대해 $z$의 이론상 distribution과 실제 $z$의 distibution의 차이를 의미
- $\mathcal{L}\_{NVAE}(\theta, \gamma)$는 모든 case에 대해 두 값의 차이를 길이만큼 나누어 준 뒤 더한 것임

$$
\begin{align}
&\mathcal{L}\_{NVAE}(\theta, \gamma)=\Sigma_{n=1}^N {1\over\Sigma_d L_{dn}}(\mathbb{E}\_{q_\gamma(z|r,s_n)}[\text{log}p_\theta(x_n|z,t_n)\] - D_{KL}(q_\gamma(z|r,s_n)||p(z))) \\
&D_{KL}(q_\gamma(z|r,s_n)||p(z)) = \Sigma_{i=1}^K D_{KL}(q_\gamma(z_i|r,s_n)||p(z_i)) \\
&\text{log}p_\theta(x_n|z,t_n) = \Sigma_{d=1}^D\Sigma_{j=1}^{L_{dn}} \text{log}p_\theta (x_{jdn}|z, t_{jdn}) \end{align}$$

**Supervised Learning**

- unsupervised learning loss에 classification loss를 더한 형태
   
$$
\begin{align}
&\mathcal{L}\_{supervised}(\theta,\gamma,\delta)= \mathcal{L}\_{NVAE}+ \lambda \mathbb{E}\_{q_\gamma(z|r,s_n)} \text{log}p_\delta (y_n|z) \\
&y^{*} = \underset{y \in \mathcal{Y}}{\text{argmax}}\mathbb{E}\_{q\_\gamma(z|r,s)}[\text{log}p_\delta (y|z)] \end{align}$$

# 5. Experiments

Datasets은 PhysioNet, MIMIC-III, human activity 이 세가지를 사용했으며 전부 sparse하고 multivariate, irregularly sampled인 time series 데이터들이다. 비교 지표는 interpolation에 대해서는 MSE, classification에 대해서는 AUC와 accuracy를 사용했으며 기존의 interpolation 관련 방법들 혹은 time series 처리 방법들과 성능을 비교하였다.


![캡쳐5](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/47f98fc9-166c-4aeb-bba8-6ed9a45f8498)

![캡쳐6](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/3ec3e71f-27ae-4a48-8b9d-db13c0c04bd1)
