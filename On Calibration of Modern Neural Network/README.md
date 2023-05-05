# On Calibration of Modern Neural Networks (2017)
논문 : https://arxiv.org/pdf/1706.04599.pdf <br>
참고자료 :  https://scikit-learn.org/stable/modules/calibration.html#calibration

Confidence calibration : the problem of predicting probability estimates representative of the true correctness likelihood

## Introduction
intro 내용 어쩌구, + calibation은 언제 제시된 개념이고~~

## Definition


$$
\begin{align*}
& X \in \mathcal{X}\text{ : input} \\
& Y \in \mathcal{Y} = {1,....,K}\text{ : label} \\
& \text{Let h be a neural network with } h(X) = (\hat{Y}, \hat{P}) \text{, where } \hat{Y}\text{ is a class prediction and }\hat{P}\text{ is its associated confidence} \\
\end{align*}
$$

$$$$

$$
\begin{align*}
& \text{Perfect Calibration : } \mathbb{P}(\mathcal{Y} = Y|\mathcal{P} = p) = p, \forall{p} \in [0,1] \\
\end{align*}
$$

ECE, MCE

## Oberserving Miscalibration
- Model capacity
- Batch Normalization
- Weight decay
- NLL

## Calibaration Methods
### For Binary
- Histogram binning
- Isotonic regression
- Bayesian Binning into Quantiles (BBQ)
- Plat scailing

### Extension to Multiclass
- Extension of binning methods
- Matrix and vector scailing

## Result

## Conclusion
