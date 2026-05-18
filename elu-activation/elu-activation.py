def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    y = []
    for i in x:
        if i>0:
            y.append(i)
        else:
            y.append(alpha*(math.exp(i) - 1))

    return y
'''
NOTES - 
ELU Activation Function Notes

--------------------------------------------------
ReLU's Two Problems
--------------------------------------------------

ReLU is defined as:

ReLU(x) = max(0, x)

ReLU became very popular in deep learning, but it has two main problems.

1. Dead Neurons

For negative inputs:

ReLU output = 0
ReLU gradient = 0

If a neuron's input stays negative for all training examples, then:

- The neuron always outputs 0
- The gradient is always 0
- The weights never update
- The neuron becomes permanently inactive

This is called the dead neuron problem.

2. Non-Zero Mean Outputs

ReLU only outputs zero or positive values.

So the average output of ReLU is usually positive.

This creates a bias shift in deeper layers.

Because of this, the next layer has to adjust for this positive shift.

This can slow down learning.

--------------------------------------------------
Leaky ReLU Limitation
--------------------------------------------------

Leaky ReLU fixes the dead neuron problem by allowing a small slope for
negative inputs.

For negative inputs:

LeakyReLU(x) = alpha * x

This gives a nonzero gradient for negative values.

However, the negative side of Leaky ReLU is just a straight line.

So it does not fully solve the mean shift problem.

--------------------------------------------------
ELU: Exponential Linear Unit
--------------------------------------------------

ELU stands for Exponential Linear Unit.

ELU modifies the negative side using an exponential curve.

For positive inputs, ELU behaves like ReLU.

For negative inputs, ELU smoothly approaches a negative saturation value.

ELU is defined as:

ELU(x) = x,                    if x > 0
ELU(x) = alpha * (e^x - 1),    if x <= 0

Here, alpha controls how negative the output can become.

--------------------------------------------------
Examples of ELU
--------------------------------------------------

Let alpha = 1.0.

For positive input:

ELU(2.0) = 2.0

For zero:

ELU(0) = 1.0 * (e^0 - 1)
       = 1.0 * (1 - 1)
       = 0

For negative inputs:

ELU(-0.5) = 1.0 * (e^(-0.5) - 1)
          ≈ -0.394

ELU(-1.0) = 1.0 * (e^(-1.0) - 1)
          ≈ -0.632

ELU(-5.0) = 1.0 * (e^(-5.0) - 1)
          ≈ -0.993

As x approaches negative infinity:

ELU(x) approaches -alpha.

If alpha = 1.0:

ELU(-infinity) approaches -1.0

So the negative side smoothly curves from 0 toward -alpha.

--------------------------------------------------
Why the Exponential Shape Helps
--------------------------------------------------

The exponential negative side gives ELU three main benefits.

--------------------------------------------------
1. Mean Activations Closer to Zero
--------------------------------------------------

ReLU only outputs non-negative values.

So its average activation is usually positive.

ELU outputs negative values for negative inputs.

Because of this, the average activation of a layer becomes closer to zero.

This helps because:

- Zero-mean activations often converge faster
- The next layer does not need to compensate for a large positive bias
- It reduces the bias shift problem
- It gives an effect similar to mean-centering

This is one reason ELU can work better than ReLU in some deep networks,
especially when batch normalization is not used.

--------------------------------------------------
2. No Dead Neurons
--------------------------------------------------

For negative inputs, ELU still has a nonzero gradient.

For x <= 0:

ELU(x) = alpha * (e^x - 1)

Derivative:

d/dx ELU(x) = alpha * e^x

Since e^x is always positive, the gradient is always positive.

So the gradient never becomes exactly zero.

This means neurons can still update even when their inputs are negative.

Therefore, ELU helps prevent dead neurons.

--------------------------------------------------
3. Smooth Transition at Zero
--------------------------------------------------

ReLU has a sharp corner at x = 0.

ELU is smooth and differentiable at x = 0.

This means:

- There is no sudden jump in gradient
- Optimization can become more stable
- The activation curve transitions smoothly from negative to positive side

Smooth gradients can help gradient-based optimization.

--------------------------------------------------
The Alpha Parameter
--------------------------------------------------

Alpha controls the negative saturation value of ELU.

As x approaches negative infinity:

ELU(x) approaches -alpha

So alpha directly sets the lower limit of the output.

Common values:

1. alpha = 1.0

This is the most common choice.

The output saturates at -1.

2. alpha = 0.5

The output saturates at -0.5.

This gives less negative output and a weaker mean-centering effect.

3. alpha = 2.0

The output saturates at -2.

This gives stronger negative output and a stronger mean-centering effect.

--------------------------------------------------
ELU vs ReLU vs Leaky ReLU
--------------------------------------------------

Consider x = -1.

For ReLU:

Output = 0
Gradient = 0

For Leaky ReLU with alpha = 0.01:

Output = -0.01
Gradient = 0.01

For ELU with alpha = 1.0:

Output = e^(-1) - 1
       ≈ -0.632

Gradient = e^(-1)
         ≈ 0.368

--------------------------------------------------
Key Differences
--------------------------------------------------

ReLU:

- Fastest to compute
- Very simple
- Can suffer from dead neurons
- Outputs are not zero-centered

Leaky ReLU:

- Prevents dead neurons
- Easy to compute
- Allows small negative values
- Negative side is still a straight line
- Does not strongly solve mean shift

ELU:

- Prevents dead neurons
- Produces negative outputs
- Pushes mean activations closer to zero
- Smooth at x = 0
- More expensive because it uses e^x

--------------------------------------------------
Where ELU Is Used
--------------------------------------------------

1. Deep Fully Connected Networks

ELU can work well in deep fully connected networks.

It is especially useful when batch normalization is not used.

This is because ELU naturally pushes activations closer to zero mean.

2. SELU

SELU stands for Scaled Exponential Linear Unit.

SELU is based on ELU but multiplies it by a special scaling constant.

It is designed for self-normalizing neural networks.

3. Image Recognition

ELU has been used in image recognition tasks.

It has shown competitive or better performance than ReLU in some benchmarks.

4. Smooth Gradient Applications

ELU is useful when smooth gradients are important.

Since ELU is differentiable at zero, it avoids the sharp corner found in
ReLU.

--------------------------------------------------
Summary
--------------------------------------------------

ReLU has two main problems:

1. Dead neurons
2. Non-zero mean outputs

Leaky ReLU fixes the dead neuron problem by allowing a small negative slope.

But Leaky ReLU does not fully solve the mean shift issue.

ELU uses an exponential curve for negative inputs.

ELU is defined as:

ELU(x) = x,                 if x > 0
ELU(x) = alpha*(e^x - 1),   if x <= 0

Benefits of ELU:

- Allows negative outputs
- Pushes mean activations closer to zero
- Helps reduce bias shift
- Prevents dead neurons
- Has smooth transition at zero

For negative inputs, ELU gradient is:

alpha * e^x

This is always positive, so gradients continue to flow.

Main drawback:

ELU is slower than ReLU and Leaky ReLU because it requires computing e^x.

ELU is useful in deep networks, especially when batch normalization is not
used.

'''