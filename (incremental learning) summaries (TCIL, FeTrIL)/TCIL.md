# TCIL
(latex 수식 에러가 있음)

## Pipeline

at step $t\in\{2,...,T\}$,

Previous extractors $\{F_1,...,F_{t-1}\}$ and previous classifier $g_{t-1}$ are frozen.

Given an image $x$ from the seen batches $\{1,...,t\}$

$u_t = [F_1(t),…,F_{t}(x)]$

The new feature extractor $F_t$ learns from both $D_t$, the $t$-th data batch and $u_{t-1}$, the feature representation at step $t-1$, by using the proposed **feature-level distillation**.

$f_t = A_t(u_t)$ 

$f$ : feature,   $A$ : attention-based feature fusion module

$o_{rt}(x) = \mathcal{W}_t(o_t) =\mathcal{W}_t(g_t(f_t))$

$o$ : output logit,  $g$ : classifier,  $\mathcal{W}$ : re-scoring module (CR)

## MLKD (Multi-level Knowledge Distillation)

$L_{kd} = \lambda\Sigma_{x\in R}L_f(x) +\mu\Sigma_{x\in R \cup D_t}L_l(x)$

$R$ : rehearsal memory set

($\lambda=0$ means the non-rehearsal setting, i.e., all loss is from $\mathcal{L}_l$)

$\mathcal{L}_f(x) = \lVert F_t(x) - F_i(x) \rVert_2$
 
$\mathcal{L}_l(x) = \Sigma_{c=1}^{\tilde{C}_{t-1}}q_c(x)(log{q_c(x) \over \hat{q}_c(x)})$

$q_c, \hat{q}_c$ : softmax with temperature,  $\hat{q}_c$ is from logit of $g_{t-1}$ and $q_t$ is from logit of $g_t$ 


## Classifier Re-scoring (CR)

$n_{old} = (\lVert w_1 \rVert, ..., \lVert w_{\hat{c}_{t-1}}\rVert)$

$n_{new} = (\lVert w_{\hat{c}_{t-1}+1}\rVert,...\lVert w_{\hat{c}_{t}} \rVert)$

$\gamma = \text{Mean}(n_{old})/\text{Mean}(n_{new})$

$o_{rt}(x) = \mathcal{W_t} (o_t) = (o_{old}(x), \gamma \cdot o_{new}(x))$

## Feature Fusion Module (FFM)

$A_c \in \mathbb{R}^{C \times 1 \times 1}$ : 1D attention map (channel attention)

$A_s \in \mathbb{R}^{1 \times H \times W}$ : 2D attention map (spatial attention) 

for feature map $u_t \in \mathbb{R}^{C \times H \times W}$

$f_t = A_t(u_t) = A_s(A_c(u_t) \otimes u_t) \otimes (A_c(u_t) \otimes u_t)$ : feature fusion process

$A_c(u_t) = \sigma(MLP(\text{AvgPool}(u_t)) + MLP(\text{MaxPool}(u_t)))$

$\text{AvgPool}(u_t), \text{MaxPool}(u_t) \in \mathbb{R}$

$A_s(u_t) = \sigma(f^{7 \times 7}([\text{AvgPool}(u_t);\text{MaxPool}(u_t)]))$

($A_c$와 $A_s$에서의 pooling은 output shape이 다름)

$f^{7\times 7}$ : convolution with kernel size $7 \times 7$

## Training Loss

$\mathcal{L} = \mathcal{L}_{clf} + \alpha \mathcal{L}_{kd} + \alpha \mathcal{L}_{div}$
