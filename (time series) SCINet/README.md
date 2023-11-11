# SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction
36th Conference on Neural Information Processing Systemp (NeurlPS 2022)<br>
분야 : Time Series Forcasting (TSF) <br>
논문 : https://arxiv.org/pdf/2106.09305.pdf <br>
github : https://github.com/cure-lab/SCINet


# Abstract
- time series 데이터릐 독특한 성질 중 한가지는 temporal relations (시간적 관계)가 두 개의 sub-sequence로 다운샘플링된 후에도 보존된다는 점이다. 
- 이러한 이점을 살려 해당 논문은 temporal modeling과 forecasting을 위해 sample convolution과 interaction을 수행하는 novel한 neural network, **SCINet**를 제안했다.
- SCINet은 recursive한 downsampling-convolve-interact architecture이다.
- 실험결과들은 SCINet이 convolution 모델들과 transformer-based 모델들에 비해 눈에 띄는 forecasting accuracy improvements를 달성했다.

  
# 1. Introduction
Time series forecasting (TSF)은 healthcare, energy management, traffic flow, financial investment 등 다양한 분야에서 중요한 역할을 수행한다.
- 해당 논문에서 제안한 SCINet은 복잡한 시간 역학이 있는 계층적 downsample-convolve-interact TSF framework이다. 여러 시간적 해상도에서 정보를 반복적으로 추출하고 exchange해가며 예측 가능성이 향상된 효과적인 representation을 학습한다. (상대적으로 낮은 permutation entropy (PE)를 통해 확인할 수 있음)
- SCI-Block이라는 것을 제안했는데 input data를 두 개의 subsequence로 downsampling하고 각 subsequence에 대해 다른 convolutional filter를 이용해 feature를 추출한다. downsampling 과정에서 손실되는 정보를 보완하기 위해 두 convolutional features 사이에 interactive learning을 추가하였다.


# 2. Related Work and Motivation
Time series forecasting은 다음과 같이 정의된다. 
- long time series $X*$와 고정된 길이 $T$를 갖는 window가 있다고 하자.
- Time series forecasting은 timestamp $t$에 대해 $\hat{X}\_{t+1:t+\tau} = (x_{t+1},...,x_{t+\tau})$를 $X\_{t-T+1:t} = (x\_{t-T+1},...,x\_t)$의 과거 $T$ steps의 데이터를 이용해 예측하는 것이다.
- $\tau$는 length of the forecast horizon이며, $x_t \in \mathbb{R}^d$는 time step $t$에서의 value이고 $d$는 variates의 수이다.

기타 선행 연구는 생략


![캡쳐7](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/71161085-9fde-4cd7-8418-9428ca9500e0)
# 3. SCINet: Sample Convolution and Interaction Network
SCINet은 encoder-decoder 구조를 갖는다. 
- encoder는 계층적인 convolutional network로서 dynamic temporal dependencies를 여러 해상도에서 다양한 convolutional filter의 set로 포착하는 역할을 한다.
- 위 그림의 Fig 2(a)에 SCI-Block의 동작이 나오는데 input data 혹은 input feature를 두 개의 sub-sequence로 downsampling한다.
- downsample된 두 subsequence 각각에 대해 convolution filter로 처리해 각 부분에서 distinct하면서도 valuable한 시간적 feature를 추출한다. downsampling 과정에서 손실될 수 있는 정보에 대해 두 subseqeunce 사이에 interactive learning을 추가시켜 보완하도록 한다.
- SCINet은 여러 개의 SCI-Blocks를 Fig 2(b)의 binary tree 구조로 배치하여 구성하였다.
- 이러한 구조의 장점은 각 SCI-Block이 전체 시계열에 대해 local과 global view를 모두 가질 수 있다는 점이다. 그럼으로써 유용한 temporal features를 추출해낼 수 있다.
- 모든 downsample-convolve-interact operation이 끝난 후, fully connected 층을 decoder로써 사용해 extracted된 features를 새 sequence로 재배치하고, original time series에 더하도록 한다.
- 복잡한 temporal pattern을 더 잘 추출하기 위해서 여러 개의 SCINet을 stack하였고, intermediate supervision을 적용하였다. (Fig 2(c))

## 3.1 SCI-Block
input feature $F$를 spliting and interactive-learning 연산을 통해 $F_{oldd}'$와 $F_{even}'$로 분해하는 역할을 한다. 
- Splitting procedure은 original sequence $F$를 odd elements인 $F_{odd}$와 even element인 $F_{even}$으로 분리하는데 각각은 temporal 해상도가 더 낮지만 원본 sequence의 대부분의 정보를 보존한다.
- 이어서 $F_{odd}$와 $F_{even}$에 대해 다른 convolutional kernels ($\phi, \psi$)를 적용해 feature를 추출한다. kernel들은 seperate되어있고 추출된 features는 distinct하면서도 valueable한 temporal relations를 조금 더 enhanced representation capabilies로 포함한다. downsampling 중 손실될 수 있는 정보는 interactive learning strategy로 보완하는데 두 subsequence 사이에서 affine transformation parameter를 학습하는 방식이다.

$$
\begin{align} F_{odd}^s = F_{odd} \odot exp(\phi(F_{even})),& ~~~~F_{even}^s =  F_{even} \odot exp(\psi(F_{odd})) \\
F_{odd}' = F_{odd}^s \pm \rho(F_{even}^s),& ~~~~F_{even}' = F_{even}^s \pm \eta(F_{odd}^s)
\end{align}
$$

- 여기서 $\odot$은 Hadamard product 혹은 element-wise production이다. 상단의 수식은 scaling transformation을 수행하는 것으로도 볼 수 있으나 scaling factors는 다른 neural network module을 통해서 학습된다.  $\phi, \psi$ 뿐 아니라 $\rho, \eta$ 역시 1D convolutional module이며 서로 다른 hidden states로 projection하는 역할을 한다.

**appendix. C** : $\phi, \psi, \rho, \eta$는 같은 network architecture이다. 첫번째로 replication padding이 적용되어 convolution operation으로 크기가 줄어드는 것을 방지한다. 1d-convolution layer with kernel size $k$이 input channel $C$를 $h\*C$로 만드는데 적용되고 이어서 LeakyRelu와 Dropout이 적용된다. $h$는 hidden size의 scale을 의미한다. 이어서 1d-convolution layer with kernel size $k$이 channel $h\*C$를  원래의 input channel $C$로 되돌리는 역할을 한다. 모든 convolution의 stride는 1이다. 두번째 convolution 이후에는 Tanh activation을 사용해 [-1,1] 사이에 값을 모은다.

## 3.2 SCINet
- Fig 2(b)에 나오듯 SCINet은 여러 개의 SCI-Block을 계층적으로 구성한다.
- $l$-th level (where $l=1,...,L$)에는 $2^l$개의 SCI-Blocks가 있다. stacked SCINet의 $k$-th SCINet 안에서 input인 time series $X$ (for $k=1$) 혹은 feature vecotr $\hat{X}^{k-1} = (\hat{x}\_{1}^{k-1},...,\hat{x}\_{\tau}^{k-1})$ (for $k>1$)는 SCI-Block을 통해 점점 다른 level에서 downsampled and processed되며 이러한 과정으로 effective feature learning of different temporal resolution이 되도록 한다. 특히 이전 level에서의 정보가 점점 accumulated되는데 즉, deep한 level에서의 feature는 shallow level에서 전달받아 추가적인 미세한 scale의 시간적(temporal) 정보를 포함하게 되는 것이다.
- $L$ level의 SCI-Block을 거친 후, odd-even splitting 연산을 뒤집고 하나의 새 sequence representation이 되도록 concatenate하면서 sub-features의 모든 elements를 재배열한다.
- 재배열된 elements에는 original time series가 residual connection을 통해 더해져 더 enhanced된 predictability를 가진 새 sequence를 생성해낸다.
- 마지막으로 간단한 fully-connected network가 enhanced sequence representation을 $\hat{X}^k = (\hat{x}_1^k,...,\hat{x}_{tau}^k)$로 decode한다.

## 3.3 Stacked SCINet
- 충분한 training sample이 있을 때, SCINet의 $K$개의 layer를 쌓아 더 나은 forecasting accuracy를 달성할 수 있다.
- 특히 SCINet의 각 output에 대한 intermediate supervision을 통해 중간의 temporal features를 학습하는 것을 쉽게 만든다.
- $k$-th intermediate SCINet의 output $\hat{X}\_k$ with length $\tau$는 original input의 part $X\_{t-(T-\tau)+1:t}$와 concatenate되어 original input으로 길이를 복구하고 ($t$시점을 기준으로 가장 late한 시계열 길이 $\tau$를 이용해 $t$시점 이후를 예측) k+1번째 SCINet에 입력으로 공급된다. 여기서 $k=1,...,K-1$이고, $K$는 stack 구조에 있는 SCINet의 총 개수이다.
- $K$-th SCINet의 output $\hat{X}^K$는 최종적인 forecasting results이다.

## 3.4 Loss Function
$k$-th SCINet의 output과 ground-truth horizontal window 사이에서의 L1-loss를 사용

$L_k = {1 \over \tau} \Sigma_{i=0}^{\tau} \lVert \hat{x}_i^k - x_i \rVert$

stacked SCINet의 total loss는 다음과 같다.

$L = \Sigma_{k=1}^K L_k$

## 3.5 Complexity Analysis
생략

# 4. Experiments
## 4.1 Datasets
![캡쳐8](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/56414115-e48e-4b8d-870f-0f23264df7ba)

총 11가지의 유명한 데이터셋으로 실험하였고 간단한 description은 table1과 같다. 모든 실험들은 multi-variate TSF setting 하에서 수행되었다. (ETTh1 데이터셋의 경우 어떻게 생겼는지 코드를 첨부해놓음)

## 4.2 Results and Analyses
168개의 input length를 사용, $\tau$는 forecast different future horizons. <br>
TSF 알고리즘에 따른 비교는 다음과 같다.

**Short-term Time Series Forecasting**
![캡쳐9](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/f149a21a-e1b9-403e-9261-3712e0f28c0b)

**Long-term Time Series Forecasting**
![캡쳐10](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/0ab61348-929a-436a-8bc6-2301642e8950)

**Multivariate Time-series Forecasting on ETF**
![캡쳐11](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/07050e92-4b4a-414e-a413-3865f1a03b9e)

다음 Fig 3은 ETTh1 데이터셋에서 랜덤하게 선택된 sequence의 예측 qualitiy를 보여준다. SCINet이 trend와 seasonality를 잘 포함하고 있음을 확인할 수 있다.

![캡쳐13](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/1ce58dff-583d-4963-a262-15bb5b3184ce)

**Univariate Time-series Forecasting on ETF**
![캡쳐12](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/49750d84-c44f-4557-9d8e-282d174dc5a1)

**Predictability estimation**
- permutation entropy (PE)를 사용해, SCINet으로 학습하여 original data의 predictability와 SCINet을 학습함으로써 enhanced된 representation을 측정하였다.
- 낮은 PE를 갖는 time series는 덜 복잡하고, 이론적으로 예측이 더 쉽다고 여겨진다. 
![캡쳐14](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/da63bf54-2581-4142-9d46-fbe95a34988f)

SCINet을 학습함으로써 향상된 representation은 original input과 비교해 더 낮은 PE를 갖는 것을 확인 할 수 있고, 이것은 미래를 예측하는게 더 쉬워짐을 의미한다.

## 4.3 Ablation studies
생략

# 5. Limitations and Future Work
- real world 상황에서 데이터에 noise가 섞여있거나 missing data, irregular time interval 문제등이 있을 수 있다. missing value가 일정 threshold를 넘어서거나 irregularly sampled된 데이터에 대해서 활용하기가 어렵다.
- SCINet은 공간적(spatial) 관계를 명시적으로 모델링하지 않지만 이러한 부분에 대한 보완이 있다면 더 나은 결과를 얻을 수 있을 것으로 기대된다.

# 6. Conclusion
generic sequence 데이터에 비해 time series 데이터가 갖는 독특한 성질로부터 motivated되어, time series modeling과 forecasting에 대한 sample convolution과 interaction network (SCINet)를 제안했다. 제안된 SCINet은 a rich set of convolutional filters을 이용한 계층적인 downsample-convolve-interact structure이다. 이 structure는 반복적으로 다른 시간적 해상도 하에서 정보를 추출하고 교환하면서 enhanced predictability를 갖는 효과적인 representation을 학습하도록 한다. 여러 TSF 데이터셋에 대해 광범위한 실험을 통해 다른 모델들에 비해 SCINet이 우수함을 증명해냈다.

