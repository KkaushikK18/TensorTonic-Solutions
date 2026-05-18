import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.array(x);
    return np.maximum(0,x)
    pass

'''
/*
Sigmoid Function Notes

The sigmoid function squashes any real number into the range (0, 1).

Formula:
sigma(x) = 1 / (1 + e^(-x))

No matter how large or small the input is, the output is always between 0 and 1.
This makes sigmoid useful for representing probabilities.

--------------------------------------------------
Shape of Sigmoid
--------------------------------------------------

The sigmoid curve is S-shaped.

For large negative inputs:
Output approaches 0.

For large positive inputs:
Output approaches 1.

At x = 0:
Output is exactly 0.5.

Some values:

sigma(-10) ≈ 0.000045
sigma(-5)  ≈ 0.0067
sigma(-2)  ≈ 0.119
sigma(-1)  ≈ 0.269
sigma(0)   = 0.5
sigma(1)   ≈ 0.731
sigma(2)   ≈ 0.881
sigma(5)   ≈ 0.9933
sigma(10)  ≈ 0.999955

--------------------------------------------------
Why Sigmoid Outputs Probabilities
--------------------------------------------------

sigma(x) = 1 / (1 + e^(-x))

1. Always positive:
   e^(-x) > 0 for all x.
   So, 1 + e^(-x) > 1.
   Therefore, sigmoid output is positive.

2. Always less than 1:
   The denominator is always greater than 1.
   So, 1 / (1 + e^(-x)) is always less than 1.

3. Monotonically increasing:
   As x increases, e^(-x) decreases.
   So the denominator decreases and the sigmoid value increases.

Because of these properties, sigmoid is used to convert raw model outputs,
called logits, into probabilities for binary classification.

--------------------------------------------------
Derivative of Sigmoid
--------------------------------------------------

The derivative of sigmoid is:

d(sigma)/dx = sigma(x) * (1 - sigma(x))

This is useful because once sigma(x) is calculated, its derivative is easy
to calculate.

Examples:

At x = 0:
sigma(0) = 0.5
derivative = 0.5 * (1 - 0.5) = 0.25

At x = 2:
sigma(2) ≈ 0.881
derivative = 0.881 * (1 - 0.881) ≈ 0.105

At x = 5:
sigma(5) ≈ 0.9933
derivative = 0.9933 * (1 - 0.9933) ≈ 0.0066

The gradient is largest at x = 0 and becomes very small when x is far
from zero.

This can cause the vanishing gradient problem in deep neural networks.

--------------------------------------------------
Numerical Stability
--------------------------------------------------

Directly computing e^(-x) can overflow for large negative x.

Example:
If x = -1000,

e^(-x) = e^1000

This is extremely large and may become infinity in a computer.

To avoid this, we use a numerically stable version.

For x >= 0:

sigma(x) = 1 / (1 + e^(-x))

For x < 0:

sigma(x) = e^x / (1 + e^x)

Both formulas are mathematically equivalent.

The second formula avoids computing e^(-x) when x is a large negative number.

--------------------------------------------------
Sigmoid vs Other Activation Functions
--------------------------------------------------

Sigmoid:
Range: (0, 1)
Gradient at 0: 0.25
Used in:
- Binary classification output layers
- Gating mechanisms

Tanh:
Range: (-1, 1)
Gradient at 0: 1.0
Used in:
- Hidden layers in older networks
- LSTM and GRU internal states

ReLU:
Range: [0, infinity)
Gradient:
- 1 for x > 0
- 0 for x < 0

Used in:
- Hidden layers in modern neural networks

Tanh is a rescaled sigmoid:

tanh(x) = 2 * sigmoid(2x) - 1

Tanh is zero-centered, meaning its output ranges from -1 to 1.
This can help neural networks converge faster than sigmoid in hidden layers.

ReLU is defined as:

ReLU(x) = max(0, x)

ReLU helps reduce the vanishing gradient problem because its gradient is 1
for positive inputs.

--------------------------------------------------
Where Sigmoid Is Used Today
--------------------------------------------------

1. Binary Classification Output Layer

A binary classifier usually outputs a single raw value called a logit.
Sigmoid converts this logit into a probability.

P(y = 1 | x) = sigmoid(logit)

Example:
If sigmoid output is 0.9, the model is confident that the class is 1.
If sigmoid output is 0.1, the model is confident that the class is 0.
If sigmoid output is 0.5, the model is uncertain.

2. Gating Mechanisms

Sigmoid is used in LSTMs and GRUs.

These gates control how much information should pass through.

Examples:
- Forget gate: decides what information to discard
- Input gate: decides what new information to store
- Output gate: decides what information to output

These gates need values between 0 and 1.

0 means block the information.
1 means allow the information to pass.
Values between 0 and 1 mean partially allow the information.

3. Attention Mechanisms

Some attention mechanisms use sigmoid instead of softmax.

Sigmoid is useful when attention weights are independent and do not need
to sum to 1.

4. Multi-label Classification

In multi-label classification, each class is independent.

Example:
An image can contain multiple labels at the same time:
- cat
- dog
- car
- person

In this case, sigmoid is applied independently to each output neuron.

This is different from softmax, where probabilities across all classes
must sum to 1.

--------------------------------------------------
Summary
--------------------------------------------------

The sigmoid function:
- Converts any real number into a value between 0 and 1
- Is useful for representing probabilities
- Has an S-shaped curve
- Gives output 0.5 when x = 0
- Has derivative sigma(x) * (1 - sigma(x))
- Can suffer from vanishing gradients
- Is commonly used in:
  - Binary classification
  - Multi-label classification
  - LSTM and GRU gates
  - Some attention mechanisms
*/
'''