import numpy as np
import math
from math import erf, sqrt
def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.array(x)
    return 0.5 * x * (1+np.vectorize(erf)(x/sqrt(2)))
    pass
'''
NOTES - 
GELU Activation Function Notes
==============================

GELU stands for Gaussian Error Linear Unit.

It is an activation function used in neural networks. It is commonly used in
modern deep learning models such as Transformers, BERT, GPT-like models, and
Vision Transformers.

GELU is a smooth activation function. It behaves somewhat like ReLU, but instead
of using a hard cutoff at zero, it uses a smooth probabilistic weighting.

------------------------------------------------------------
Mathematical Formula
------------------------------------------------------------

GELU(x) = (1/2) * x * (1 + erf(x / sqrt(2)))

where:

erf = error function
sqrt(2) = square root of 2

The same formula can also be written as:

GELU(x) = x * Phi(x)

where:

Phi(x) = standard normal cumulative distribution function (CDF)

Phi(x) represents the probability that a standard Gaussian random variable Z is
less than or equal to x.

So:

Phi(x) = P(Z <= x), where Z ~ N(0, 1)

Therefore:

GELU(x) = x * P(Z <= x)

------------------------------------------------------------
Meaning of the Formula
------------------------------------------------------------

The formula:

GELU(x) = (1/2) * x * (1 + erf(x / sqrt(2)))

means that GELU multiplies the input x by a probability-like value.

The term:

(1/2) * (1 + erf(x / sqrt(2)))

is equal to Phi(x), the standard normal CDF.

So GELU can be understood as:

input value * probability that a Gaussian random variable is less than input

In simple words:

GELU does not simply keep or remove an input.
It smoothly scales the input depending on its value.

------------------------------------------------------------
Intuition
------------------------------------------------------------

ReLU uses a hard decision:

if x > 0, output x
if x <= 0, output 0

But GELU uses a soft decision:

output = x * probability factor

This means:

1. Large positive values are mostly kept.
2. Large negative values are almost removed.
3. Small negative values are not immediately converted to zero.
4. The transition around zero is smooth.

------------------------------------------------------------
Behavior of GELU
------------------------------------------------------------

1. If x is large positive:

   Phi(x) is close to 1.

   GELU(x) = x * Phi(x)
           ≈ x * 1
           ≈ x

   So GELU keeps large positive values almost unchanged.

Example:

x = 3
Phi(3) ≈ 0.998

GELU(3) = 3 * 0.998
        ≈ 2.994

2. If x = 0:

   Phi(0) = 0.5

   GELU(0) = 0 * 0.5
           = 0

3. If x is negative:

   Phi(x) is less than 0.5.

Example:

x = -1
Phi(-1) ≈ 0.1587

GELU(-1) = -1 * 0.1587
         ≈ -0.1587

Unlike ReLU, GELU does not convert small negative values directly to zero.

4. If x is large negative:

   Phi(x) is close to 0.

Example:

x = -3
Phi(-3) ≈ 0.0013

GELU(-3) = -3 * 0.0013
         ≈ -0.0039

So GELU almost removes large negative values.

------------------------------------------------------------
Comparison with ReLU
------------------------------------------------------------

ReLU formula:

ReLU(x) = max(0, x)

ReLU behavior:

negative input -> 0
positive input -> same value

GELU behavior:

negative input -> small negative value
positive input -> mostly same value

Main differences:

1. ReLU has a hard cutoff at zero.
2. GELU has a smooth transition around zero.
3. ReLU completely removes negative values.
4. GELU softly reduces negative values.
5. ReLU is computationally cheaper.
6. GELU is slightly more expensive but smoother.
7. GELU is commonly used in Transformer-based models.

------------------------------------------------------------
Why GELU is Useful
------------------------------------------------------------

GELU is useful because it works like a smooth probabilistic gate.

Instead of making a hard decision like:

keep or discard

it makes a soft decision like:

how much of the input should pass?

This smooth behavior helps neural networks learn better representations in many
deep learning architectures.

GELU is especially popular in:

1. Transformer models
2. BERT
3. GPT-like models
4. NLP models
5. Vision Transformer models

------------------------------------------------------------
Approximate Outputs
------------------------------------------------------------

For input values:

x = [0, 2, -2]

GELU outputs approximately:

[0.0, 1.95449974, -0.04550026]

Explanation:

GELU(0)  = 0
GELU(2)  ≈ 1.9545
GELU(-2) ≈ -0.0455

------------------------------------------------------------
NumPy Implementation
------------------------------------------------------------

import numpy as np
from math import erf, sqrt

def gelu(x):
    x = np.array(x)
    return 0.5 * x * (1 + np.vectorize(erf)(x / sqrt(2)))

# Example
x = np.array([0, 2, -2])
print(gelu(x))

# Output:
# [ 0.          1.95449974 -0.04550026 ]

------------------------------------------------------------
Short Summary
------------------------------------------------------------

GELU means Gaussian Error Linear Unit.

Formula:

GELU(x) = (1/2) * x * (1 + erf(x / sqrt(2)))

or:

GELU(x) = x * Phi(x)

where Phi(x) is the standard normal CDF.

GELU multiplies each input by the probability that a Gaussian random variable is
less than or equal to that input.

It is a smooth version of ReLU.

For large positive inputs, GELU output is almost equal to x.
For large negative inputs, GELU output is almost zero.
For small negative inputs, GELU allows a small negative value to pass.

This makes GELU smoother and more flexible than ReLU.
'''