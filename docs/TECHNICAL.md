# 🛠 MyTorch: The Mathematical Blueprint

This document provides the formal mathematical derivation for every component in the MyTorch framework, enabling full replication of the logic.

---

## 1. Linear Layer (Fully Connected)
The fundamental building block for affine transformations.

**Forward Pass:**
$$Y = XW^T + b$$
Where $X \in \mathbb{R}^{B \times D_{in}}$, $W \in \mathbb{R}^{D_{out} \times D_{in}}$, and $b \in \mathbb{R}^{D_{out}}$.

**Backward Pass:**
* $\frac{\partial L}{\partial W} = (\frac{\partial L}{\partial Y})^T X$
* $\frac{\partial L}{\partial b} = \sum (\frac{\partial L}{\partial Y})$
* $\frac{\partial L}{\partial X} = (\frac{\partial L}{\partial Y}) W$

---

## 2. Activation Functions
### ReLU (Rectified Linear Unit)
**Logic:** $f(z) = \max(0, z)$
**Gradient:** $f'(z) = 1$ if $z > 0$ else $0$.

---

## 3. Batch Normalization (Training Mode)
Used to stabilize internal covariate shift.

**Step-by-Step Transform:**
1. **Mean:** $\mu_B = \frac{1}{B} \sum X$
2. **Variance:** $\sigma^2_B = \frac{1}{B} \sum (X - \mu_B)^2$
3. **Normalize:** $\hat{X} = \frac{X - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$
4. **Scale & Shift:** $Y = \gamma \hat{X} + \beta$

**Backpropagation:**
The gradient $\frac{\partial L}{\partial X}$ is computed using the multivariate chain rule to account for the dependency on $\mu_B$ and $\sigma^2_B$.

---

## 4. Optimization: AdamW
The "Greedy" optimizer used to reach 98.59% accuracy.

**Update Equations:**
For each parameter $\theta$ and gradient $g_t$:
1. **Momentum:** $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2. **Velocity:** $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3. **Bias Correction:** $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
4. **AdamW Update:** $\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$
*(Where $\lambda$ is the Weight Decay coefficient)*.

---

## 5. Loss Function: Label Smoothing Cross-Entropy
Prevents overconfidence by softening the target distribution.

**Target Transformation:**
Instead of a 1-hot vector, the new target $q$ for class $k$ is:
$$q_k = \begin{cases} 1 - \alpha & \text{if } k = \text{target} \\ \frac{\alpha}{K-1} & \text{otherwise} \end{cases}$$
Where $\alpha$ is the smoothing factor ($0.1$) and $K$ is the number of classes.

**Loss Calculation:**
$$L = -\sum q_k \log(p_k)$$
Where $p$ is the Softmax output.

---

## 6. Initialization: Kaiming (He)
Standard for ReLU networks to prevent vanishing/exploding gradients.
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{D_{in}}}\right)$$
