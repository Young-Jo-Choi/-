# SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction
36th Conference on Neural Information Processing Systemp (NeurlPS 2022)<br>
분야 : Time Series Forcasting (TSF)<br>
논문 : https://arxiv.org/pdf/2106.09305.pdf <br>
github : https://github.com/cure-lab/SCINet


# Abstract

# 1. Introduction

# 2. Related Work and Motivation
Time series forecasting은 다음과 같이 정의된다. 
- long time series $X*$와 고정된 길이 $T$를 갖는 window가 있다고 하자.
- Time series forecasting은 timestamp $t$에 대해 $\hat{X}\_{t+1:t+\tau} = (x_{t+1},...,x_{t+\tau})$를 $X\_{t-T+1:t} = (x\_{t-T+1},...,x\_t)$의 과거 $T$ steps의 데이터를 이용해 예측하는 것이다.
- $\tau$는 length of the forecast horizon이며, $x_t \in \mathbb{R}^d$는 time step $t$에서의 value이고 $d$는 variates의 수이다.
- ${}$ $\{{\}}$

생략
