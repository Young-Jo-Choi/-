# multi-time attention networks for ~

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

**********Notation**********

D

**Time Embedding** 

 ****

$$
\phi_h(t)[i] = \begin{cases}\omega_{0h} \cdot t+\alpha_{0h},~~~~ \qquad if \quad i=0 &\\sin(\omega_{ih} \cdot t +\alpha_{ih}), \quad if \quad 0<i<d_r \end{cases}
$$

$$
\begin{align} 
	\text{mTAN}(t,\mathbf{s})[j] & = \sum_{h=1}^H \sum_{d=1}^D \hat{x}_{hd}(t,\mathbf{s})\cdot U_{hdj}\\ 
    	\hat{x}_{hd}(t,\mathbf{s}) & =\sum_{i=1}^{L_d} \kappa_h (t,t_{id})x_{id}\\ 
        \kappa_h (t,t_{id}) & = {{\text{exp}(\phi_h(t)\mathbf{wv}^T \phi_h(t_{id})^T /\sqrt{d_k}} \over {\Sigma_{i'=1}^{L_d} \text{exp}(\phi_h(t)\mathbf{wv}^T \phi_h(t_{i'd})^T /\sqrt{d_k}}} 
\end{align}
$$

# 4. Encoder-Decoder framework

**Encoder**

$$
\begin{align} z_k &\sim p(z_k) \\h_{RNN}^{dec}&= \text{RNN}^{dec}(z)\\ h_{TAN}^{dec}&= \text{mTAND}^{dec}(t,h_{RNN}^{dec}) \\ x_{id}&\sim N(x_{id};f^{dec}(h_{i,TAN}^{dec})[d], \sigma^2 I \end{align}
$$

**Decoder**

$$
\begin{align} h_{TAN}^{enc} &= \text{mTAND}^{enc}(r,s)\\h_{RNN}^{enc} &=\text{RNN}^{enc}(h_{TAN}^{enc}) \\z_k \sim q_\gamma(z_k|\mu_k,{\sigma}^2_k), \; \mu_k &= f_{\mu}^{enc}(h_{k,RNN}^enc), \; {\sigma}^2_k =\text{exp}(f_\sigma^{enc}(h_{k,RNN}^{enc}))\end{align} 
$$

******************************************Unsupervised Learning******************************************

$\mathcal{L}_{NVAE}(\theta, \gamma)=\sum_{n=1}^N {1\over\Sigma_d L_{dn}}(\mathbb{E}_{q_\gamma(z|r,s_n)}[\text{log}p_\theta(x_n|z,t_n)] - D_{KL}(q_\gamma(z|r,s_n)||p(z)))$

$D_{KL}(q_\gamma(z|r,s_n)||p(z)) = \sum_{i=1}^K D_{KL}(q_\gamma(z_i|r,s_n)||p(z_i))$

$\text{log}p_\theta(x_n|z,t_n) = \sum_{d=1}^D\sum_{j=1}^{L_{dn}} \text{log}p_\theta (x_{jdn}|z, t_{jdn})$

**************************************Supervised Learning**************************************

$\mathcal{L}_{supervised}(\theta,\gamma,\delta)= \mathcal{L}_{NVAE}+ \lambda \mathbb{E}_{q_\gamma(z|r,s_n)} \text{log}p_\delta (y_n|z)$

$y^{*} = \underset{y \in \mathcal{Y}}{\text{argmax}}\mathbb{E}_{q_\gamma(z|r,s)}[\text{log}p_\delta (y|z)]$