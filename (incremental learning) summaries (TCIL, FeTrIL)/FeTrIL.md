# FeTrIL

We introduce a method which combines a fixed feature extractor and a pseudo-features generator to improve the stability-plasticity balance. The generator uses a simple yet effective geometric translation of new class features to create representations of past classes, made of pseudo-features. The translation of features only requires the storage of the **centroid** representations of past classes to produce their **pseudo-features**.

New classes are represented by their image features obtained from the feature extractor. Past classes are represented by pseudofeatures which are derived from features of new classes by using a **geometric translation process. (This translation moves features toward a region of the features space which is relevant for past classes.)**

$\hat{f}^t(C_p)$ : pseudo-features of past classes

$\mathcal{G}$ : generator of pseudo feature. 

$\hat{f}^t(C_p) = f(c_n) + \mu(C_p) - \mu(C_n)$

$C_p$ : target past class for which pseudo-features are needed

$C_n$ : new classes for which images b are available

$f(c_n)$ : features of a sample $c_n$ of class $C_n$ extracted with $\mathcal{F}$

$\mu(C_p), \mu(C_n)$ : mean features of classes $C_p$ and $C_n$ extracted with $\mathcal{F}$

$\hat{f}^t(C_p)$ : pseudo-features vector of a pseudo-sample $c_p$ of class $C_p$ produced in the $t^{th}$ incremental state

## overall process

![Untitled](FeTrIL%2010a874b9677747449b9054b5e870df6b/Untitled.png)

## selection of pseudo features

strategies

- $\text{FeTrIL}^k$ : $s$ features are transferred from the $k^{th}$ similar new class of each class $C_p$. Similarities between the target $C_p$ and $C_n$ is computed using the cosine similarity between the centroids of each pair of classes.
- $\text{FeTrIL}^{rand}$ : $s$ features are randomly selected from all new classes.
- $\text{FeTrIL}^{herd}$ : $s$ features are selected from any new class based on a herding algorithm.

(pseudo-feature 자체는 new class 내의 여러 이미지 샘플로부터 여러 개 생성 가능)

## Linear classification layer training

$\mathcal{W}^t = \{w^t(C_1),...,w^t(C_P),w^t(C_{P+1}),...w^t(C_{P+N}) \}$

$P$ : past classes

$N$ : new classes

$w^t$ : the weight of known classes in the $t^{th}$ CIL state

$\mathcal{W}^t$ can be implemented using different classifiers : fully-connected layer and LinearSVCs