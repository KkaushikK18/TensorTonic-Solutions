import numpy as np

def sigmoid(x):
    x = np.array(x)
    return 1/(1+np.exp(-x))
    pass

'''
NOTES - 
# Sigmoid Function

## What the Sigmoid Function Does

The sigmoid function squashes any real number into the range **(0, 1)**:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

No matter how large or small the input is, the output is always between **0 and 1**.

This makes sigmoid useful for representing probabilities.

---

## Shape of the Sigmoid Function

The sigmoid curve has an **S-shape**. The word *sigmoid* comes from the Greek letter sigma, which looks like an S.

### Behavior

- For large negative inputs, the output approaches **0**
- For large positive inputs, the output approaches **1**
- At \(x = 0\), the output is exactly **0.5**

### Some Concrete Values

| Input \(x\) | Sigmoid Output \(\sigma(x)\) |
|---:|---:|
| -10 | 0.000045 |
| -5 | 0.0067 |
| -2 | 0.119 |
| -1 | 0.269 |
| 0 | 0.5 |
| 1 | 0.731 |
| 2 | 0.881 |
| 5 | 0.9933 |
| 10 | 0.999955 |

---

## Why Sigmoid Outputs Probabilities

The formula is:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

This guarantees the output is always between **0 and 1**.

### 1. Always Positive

Since:

\[
e^{-x} > 0
\]

for all real values of \(x\), the denominator is:

\[
1 + e^{-x} > 1
\]

So the output is always positive.

---

### 2. Always Less Than 1

The denominator is always greater than 1:

\[
1 + e^{-x} > 1
\]

Therefore:

\[
\frac{1}{1 + e^{-x}} < 1
\]

So sigmoid always gives a value less than 1.

---

### 3. Monotonically Increasing

As \(x\) increases, \(e^{-x}\) decreases.

So the denominator becomes smaller, and the value of sigmoid increases.

Therefore, sigmoid is a monotonically increasing function.

These properties make sigmoid useful for converting raw model outputs, called **logits**, into probabilities for binary classification.

---

## Derivative of Sigmoid

The derivative of sigmoid has a simple form:

\[
\frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))
\]

This means that once we compute \(\sigma(x)\), the gradient is easy to calculate.

---

## Derivative Values

### At \(x = 0\)

\[
\sigma(0) = 0.5
\]

\[
\sigma'(0) = 0.5(1 - 0.5)
\]

\[
\sigma'(0) = 0.25
\]

This is the maximum derivative value of sigmoid.

---

### At \(x = 2\)

\[
\sigma(2) \approx 0.881
\]

\[
\sigma'(2) = 0.881(1 - 0.881)
\]

\[
\sigma'(2) \approx 0.105
\]

---

### At \(x = 5\)

\[
\sigma(5) \approx 0.9933
\]

\[
\sigma'(5) = 0.9933(1 - 0.9933)
\]

\[
\sigma'(5) \approx 0.0066
\]

The gradient is largest at \(x = 0\) and quickly becomes very small as \(x\) moves far away from zero.

This can cause the **vanishing gradient problem** in deep neural networks.

---

## Numerical Stability

Computing \(e^{-x}\) directly can cause overflow for large negative values of \(x\).

For example:

\[
e^{-(-1000)} = e^{1000}
\]

This value is extremely large and may become infinity in a computer.

To avoid this, we use a numerically stable implementation.

---

## Stable Sigmoid Formula

### For \(x \ge 0\)

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

### For \(x < 0\)

\[
\sigma(x) = \frac{e^x}{1 + e^x}
\]

Both formulas are mathematically equivalent.

However, the second formula avoids computing \(e^{-x}\) when \(x\) is a large negative number.

---

## Sigmoid vs Other Activation Functions

| Activation | Range | Gradient Behavior | Common Use |
|---|---|---|---|
| Sigmoid | \((0, 1)\) | Maximum gradient is 0.25 | Binary classification output, gates |
| Tanh | \((-1, 1)\) | Gradient at 0 is 1.0 | Hidden layers, LSTM/GRU internal states |
| ReLU | \([0, \infty)\) | Gradient is 1 for \(x > 0\), 0 for \(x < 0\) | Hidden layers in modern neural networks |

---

## Tanh and Sigmoid Relationship

Tanh is a rescaled version of sigmoid:

\[
\tanh(x) = 2\sigma(2x) - 1
\]

Tanh is zero-centered because its output range is from **-1 to 1**.

This often helps neural networks converge faster than sigmoid when used in hidden layers.

---

## ReLU Compared to Sigmoid

ReLU is defined as:

\[
\text{ReLU}(x) = \max(0, x)
\]

ReLU helped reduce the vanishing gradient problem in deep networks.

Its gradient is:

- 1 for \(x > 0\)
- 0 for \(x < 0\)

Because gradients do not shrink in the positive region, ReLU became the default activation function for hidden layers in many modern neural networks.

---

## Where Sigmoid Is Used Today

## 1. Binary Classification Output Layer

The final layer of a binary classifier often outputs a single raw value called a **logit**.

Sigmoid converts this logit into a probability:

\[
P(y = 1 \mid x) = \sigma(\text{logit})
\]

Example:

- If sigmoid output is 0.9, the model is highly confident that the class is 1
- If sigmoid output is 0.1, the model is highly confident that the class is 0
- If sigmoid output is 0.5, the model is uncertain

---

## 2. Gating Mechanisms

Sigmoid is used in gates inside LSTMs and GRUs.

These gates control how much information should pass through.

Examples:

- **Forget gate:** decides what information to discard from the cell state
- **Input gate:** decides what new information to store
- **Output gate:** decides what information to output

These gates need values between **0 and 1**.

They act like soft switches:

- 0 means block the information
- 1 means allow the information to pass
- Values between 0 and 1 mean partially allow the information

---

## 3. Attention Mechanisms

Some attention mechanisms use sigmoid instead of softmax.

Sigmoid is useful when attention weights should be independent and do not need to sum to 1.

---

## 4. Multi-Label Classification

In multi-label classification, each class is independent.

For example, an image can contain multiple labels at the same time:

- cat
- dog
- car
- person

In this case, sigmoid is applied independently to each output neuron.

This is different from softmax, where the probabilities across classes must sum to 1.

---

# Summary

The sigmoid function:

- Converts any real number into a value between 0 and 1
- Is useful for representing probabilities
- Has an S-shaped curve
- Outputs 0.5 when \(x = 0\)
- Has derivative:

\[
\sigma'(x) = \sigma(x)(1 - \sigma(x))
\]

- Can suffer from vanishing gradients
- Is commonly used in:
  - Binary classification
  - Multi-label classification
  - LSTM and GRU gates
  - Some attention mechanisms
'''