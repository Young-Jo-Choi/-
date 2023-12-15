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

![image](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/8d9c18b2-3752-45a9-a269-00a79a780225)

$$
\begin{align} p(y=c|x) & = \int_{\tilde{\mathfrak{D}}} p(y=c|\tilde{x}_t)p(\tilde{x}_t|x)d\tilde{x}_t \\
& \approx {1 \over A} \sum\_{t=1}^A p(y=c|\tilde{x}_t)
\end{align}
$$

$$\tilde{x} = f_r (x|\theta_r),~~r=1,...,R$$

$$\tilde{x} \sim \sum_{r=1}^R w_r*f_r (x|\theta_r)$$

$$S_c = \sum_{t=1}^T \mathbb{1}\_c \arg\max_\hat{c} p(y=\hat{c}|\tilde{x}_t)$$

$$u(x) = 1-{1 \over T}\max_c S_c$$

![image](https://github.com/Young-Jo-Choi/paper_study/assets/59189961/15cbddec-c80a-4c91-ba6f-d0e4adff5260)


### 4.2 Diversity Enhancement by Augmentation

**Mixed-Label Data Augmentation**
$$\tilde{x} = \mathbf{m} \odot x_1 + (1-\mathbf{m}) \odot x_2$$

$$\tilde{y} = \lambda y_1 + (1-\lambda)y_2$$

**Automated Data Augmentation**

## 5. Experimental Setup


