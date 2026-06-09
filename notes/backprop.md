### Page 1

Linear Classification

* image $x_i \in \mathbb{R}^D \qquad i \in \{1, 2, 3, \dots, N\}$
* label $y_i \qquad \qquad y_i \in \{1, 2, 3, \dots, K\}$
* $N$ examples (dimensionality $D$), $K$ categories
↳ $D = \text{length} \times \text{width} \times \text{rgb}$ (flattened)
* linear mapping $f: \mathbb{R}^D \rightarrow \mathbb{R}^K$
↳ raw image pixels to class scores
$f(x_i, W, b) = Wx_i + b$
↳ $x_i$: pixels flattened, vector size $D \times 1$
↳ $W$: matrix of size $K \times D$ (weights/parameters)
↳ $b$: vector, size $K \times 1$ (bias vector)
* W & b customizable
* goal: match computed scores to ground truth labels
↳ let correct class score be the max over the scores of incorrect classes

Pro: after training/learning, W & b are all that's needed
↳ entire training set can be discarded
↳ classifying involves matrix multiplication & addition (computationally faster)

* function essentially likes/dislikes certain colors (rgb) at certain positions in the image

[Diagram showing matrix W with rows labeled "red W", "green W", "blue W", multiplied by vector $x_i$, plus vector b with rows "r", "g", "b", resulting in a vector with rows "A score", "B score", "C score"]

---

### Page 2

* Graph View: each image is a point
↳ each line indicates the points in space where the category specified scores a zero
↳ arrow shows direction of increase (linearly)
[Graph diagram with intersecting lines labeled "airplanes", "cars", "deers", each with a perpendicular arrow pointing outward]
* eliminating b trick: extend x's dimension by 1
↳ before: $x \ [D, 1] \qquad$ after: $x \ [D+1, 1]$ (holds 1)
↳ before: $W \ [K, D] \qquad$ after: $W \ [K, D+1]$ (holds b)
* Loss Function: essentially how unhappy we are with the function's outcome (aka cost function or objective)
↳ high loss = bad classification
* Multiclass Support Vector Machine (SVM) loss
↳ requires the correct class to score higher than the others by some fixed margin $\Delta$.
* let $s_j = f(x_i, W)_j \leftarrow$ specific element in vector
Loss $L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta) \qquad$ ($y_i$ is correct label)
correct class scores $\Delta$ or above the incorrect ones
* hinge loss: threshold at zero $\max(0, -)$ function
* squared hinge loss (SVM): $\max(0, -)^2$

Goal:
[Number line diagram showing tick marks for "scores for other classes" on the left, a distance labeled "delta", and a tick mark for "correct class" on the right]

---

### Page 3

* Loophole! suppose W correctly classifies and $L_i = 0$ for all i
↳ thus W is not unique, ex. $\lambda W$ ($\lambda > 1$)
the difference in scores would also be scaled by $\lambda$
* Regularization Penalty $R(W)$: encodes some preference for a certain set of weights W over others (removes the ambiguity)
* Squared L2 Norm: discourages large weights through an elementwise quadratic penalty over all parameters
$R(W) = \sum_k \sum_l W_{k,l}^2$
* Multiclass Support Vector Machine Loss: 2 components, data loss (average $L_i$), regularization loss

$$L = \underbrace{\frac{1}{N} \sum_i L_i}_{\text{data loss}} + \underbrace{\lambda R(W)}_{\text{regularization loss}}$$



$N$: # of training examples
$\lambda$: hyperparameter (determined by cross-validation)
↳ including L2 penalty $\Rightarrow$ max margin property, improves generalization
↳ L2 penalty prefers smaller & more diffuse weights
$\Rightarrow$ final classifier considers all input dimensions
* due to the regularization penalty, $L \neq 0$ (only possible if $W=0$)
* hyperparameters $\Delta$ & $\lambda$ similar in function
↳ controls tradeoff of data loss vs regularization loss
↳ W directly affects the scores & the differences
$\Rightarrow \Delta = 1$ can be set without worry
* real issue: how large we allow weights to grow ($\lambda$)

---

### Page 4

* Binary Support Vector Machine:

$$L_i = C \max(0, 1 - y_i w^T x_i) + R(W) \qquad (C \propto \frac{1}{\lambda})$$


* Softmax Classifier: different loss function
↳ SVM treats $f(x_i, W)$ as uncalibrated & hard to interpret scores for each class
↳ softmax output is more intuitive, normalized class probabilities & probabilistic interpretation
* $f(x_i, W) = W x_i \leftarrow$ interpret as unnormalized log probabilities
* cross-entropy loss instead of hinge loss

$$
L_i = -\log \left( \frac{e^{f_{y_i}}}{\sum_j e^{f_j}} \right) = -f_{y_i} + \log \sum_j e^{f_j}
$$



where $f_j$ is the j-th element of the vector class scores f.
* full loss: mean of $L_i$ over $N$ plus regularization $R(W)$
* softmax function: $f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}$
↳ takes a vector of arbitrary real-valued scores (in z) and squashes it into a vector of values between 0-1 that sum to 1

---

### Page 5

* Cross-Entropy Loss
↳ between "true" distribution p & estimated distribution q:

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$



$\Rightarrow$ minimizing cross-entropy between
$\Rightarrow$ cross-entropy objective wants the predicted distribution to have all its mass on the correct answer
↳ probabilistic interpretation: $P(y_i \mid x_i ; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j}}$
* normalized probability assigned to the correct label $y_i$ given image $x_i$ & parameterized by W
* recall that softmax classifier interprets the scores vector f as the unnormalized log probabilities
$\Rightarrow$ probabilities sum to one
$\Rightarrow$ minimizing the negative log likelihood of the correct class
↳ Numeric stability: exponents $\Rightarrow$ large numbers, and large divide large can be numerically unstable

$$
\frac{e^{f_{y_i}}}{\sum_j e^{f_j}} = \frac{C e^{f_{y_i}}}{C \sum_j e^{f_j}} = \frac{e^{f_{y_i} + \log C}}{\sum_j e^{f_j + \log C}}
$$



$\log C = -\max_j f_j$ most commonly
$\Rightarrow$ shift values inside f so that largest is zero



In code:

```python
f -= np.max(f)                      // subtract
p = np.exp(f) / np.sum(np.exp(f))   // exp & norm.

```

---

### Page 6
Overview:
1) Matrix Multiply + Bias Offset

$$
\begin{bmatrix}
0.01 \& -0.05 \& 0.1 \& 0.05 \\
0.9 \& 0.2 \& 0.015 \& 0.16 \\
0.0 \& -0.45 \& -0.2 \& 0.03
\end{bmatrix}
$$

$$
\begin{aligned}
&
\begin{bmatrix}
0.01 & -0.05 & 0.1 & 0.05\\
0.9 & 0.2 & 0.015 & 0.16\\
0.0 & -0.45 & -0.2 & 0.03
\end{bmatrix}
\begin{bmatrix}
-15\\
22\\
-44\\
56
\end{bmatrix}
+
\begin{bmatrix}
0.0\\
0.2\\
-0.3
\end{bmatrix}
&=
\begin{bmatrix}
-2.85\\
0.86\\
0.28
\end{bmatrix}
\end{aligned}
$$

2a) Hinge Loss (SVM): $L = \frac{1}{N} \sum L_i + \lambda R(W)$

$$
L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)
$$

$$
\begin{bmatrix}
-2.85 \\ 0.86 \\ 0.28
\end{bmatrix}
\quad \max(0, -2.85 - 0.28 + 1) + \max(0, 0.86 - 0.28 + 1) = 1.58
$$

2b) Cross-Entropy Loss (Softmax): 

$$
L_i = -\log \left( \frac{e^{f_{y_i}}}{\sum e^{f_j}} \right)
$$

$$
\begin{bmatrix}
-2.85 \\ 0.86 \\ 0.28
\end{bmatrix}
\xrightarrow{\text{np.exp}(f)}
\begin{bmatrix}
0.058 \\ 2.36 \\ 1.32
\end{bmatrix}
\xrightarrow{\div \text{np.sum}(\text{np.exp}(f))}
\begin{bmatrix}
0.016 \\ 0.631 \\ 0.353
\end{bmatrix}
$$

$-\log(0.353) = 1.04$

* if numbers too large, `- np.max(f)` before `np.exp(f)`

* softmax provides "probabilities", but how peaky/diffuse they are depends on regularization strength $\lambda$.
    Ex: $[1, -2, 0] \rightarrow [2.71, 0.14, 1] \rightarrow [0.7, 0.04, 0.26]$
    higher $\lambda \Rightarrow$ greater penalization $\Rightarrow$ smaller W
    $[0.5, -1, 0] \rightarrow [1.65, 0.37, 1] \rightarrow [0.55, 0.12, 0.33]$ (more diffuse)
    ↳ $\uparrow \lambda \Rightarrow$ more uniform output probabilities
* SVM vs Softmax: comparable results
    ↳ but SVM can achieve $L=0$, doesn't micromanage
    ↳ but Softmax is never really happy

---

### Page 7

Optimization: Stochastic Gradient Descent

* optimization: the process of finding the set of parameters W that minimizes the loss function
* visualizing the loss function (1D)
↳ generate a random weight matrix W (point in space)
↳ move along a ray & record loss
$L(W + a W_1)$, random direction $W_1$, variable a
* visualizing the loss function (2D)
↳ $L(W + a W_1 + b W_2)$ vary a, b
* SVM cost function is a convex function (bowl-shaped)
↳ due to max operation, function has kinks
$\Rightarrow$ non-differentiable loss function
solution: use subgradient (sometimes called gradient)
* strat 1: random search (bad idea)
↳ try out random weights & keep track of the best
$\Rightarrow \sim 15.5\%$ accuracy
↳ start random, then iteratively refine to lower loss
* test code:

```python
scores = Wbest.dot(Xte_cols)
Yte_predict = np.argmax(scores, axis = 0)
np.mean(Yte_predict == Yte)

```

---

### Page 8

* strat 2: random local search
↳ start with random W and random $\delta W$, if $W + \delta W$ has a smaller loss, then update
$\Rightarrow$ classification accuracy of 21.4%
* strat 3: follow the gradient of the loss function
↳ essentially the max directional derivative $Du$
* gradient: the vector of partial derivatives in each dimension
* compute numerically with finite differences (slow but easy)
↳ computes partial derivative at each slot using formula $\frac{f(x+h) - f(x)}{h}$ but $\frac{f(x+h) - f(x-h)}{2h}$ works better
↳ `W_new = W - gradient * step_size`
* step-size: AKA learning rate, determines how far along in the determined direction we should take
↳ small $\Rightarrow$ slow but consistent progress
↳ large $\Rightarrow$ faster, but can backfire


* NOT a scalable solution (linear complexity)

---

### Page 9

$$\begin{bmatrix}
L_1 \\ L_2 \\ L_3 \\ \vdots \\ L_N
\end{bmatrix}
\qquad
[p_1, p_2 \dots p_D] = K
\begin{bmatrix}
p_{1,1} & p_{2,1} & \dots & p_{D,1} \\
p_{1,2} &  &  & \vdots \\
\vdots &  & \ddots & \vdots \\
p_{1,K} &  & \dots & p_{D,K}
\end{bmatrix}
\qquad
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\begin{bmatrix}
1 & 2 & 0 \\
4 & 5 & 6
\end{bmatrix}
= \begin{bmatrix}
9 & 12 & 15 \\
\vdots & & \vdots
\end{bmatrix}$$

* computing gradient analytically with Calculus
↳ can be more error prone forces implementing
$\Rightarrow$ gradient check: compare analytic gradient vs numerical gradient for correctness
* Ex. SVM loss function for a single data point

$$L_i = \sum_{j \neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta)$$



take gradient with respect to $w_{y_i}$:

$$\nabla_{w_{y_i}} L_i = - \left( \sum_{j \neq y_i} \mathbb{1}(w_j^T x_i - w_{y_i}^T x_i + \Delta > 0) \right) x_i$$



↳ $\mathbb{1}$ is the indicator function: true $\rightarrow 1$, false $\rightarrow 0$
* essentially multiplying $x_i$ to every term contributing to the loss function
take gradient with respect to $w_j$:

$$\nabla_{w_j} L_i = \mathbb{1}(w_j^T x_i - w_{y_i}^T x_i + \Delta > 0) x_i$$


* Gradient Descent: evaluating gradient & performing a parameter update
* vanilla code:

```python
while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += -step_size * weights_grad

```

---

### Page 10

* mini-batch gradient descent:
* large dataset $\Rightarrow$ huge computation
↳ computing full loss function for only one parameter update is wasteful
$\Rightarrow$ compute gradient over batches: vanilla code



```python
while True:
    data_batch = sample_training_data(data, 256)
    weights_grad = evaluate_gradient(
        loss_fun, data_batch, weights
    )
    weights += -step_size * weights_grad

```

* extreme case: Stochastic Gradient Descent (SGD)
↳ mini-batch contains only a single example
↳ AKA: on-line gradient descent
↳ but this term usually used to refer to the general usage of mini-batches.
* mini-batch size is a hyperparameter, BUT usually ruled by memory constraints
↳ not usually used to optimize
↳ use powers of 2 (usually works faster)
