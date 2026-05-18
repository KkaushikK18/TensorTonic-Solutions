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
Activation Functions and ReLU Notes

--------------------------------------------------
What Activation Functions Do
--------------------------------------------------

A neural network is built using layers of linear transformations.

Each layer usually does:

1. Multiply by a weight matrix
2. Add a bias

This can be written as:

y = W*x + b

The problem is that if we stack only linear layers, the final result is
still just a linear function.

Example:

Two layers without activation:

y = W2 * (W1*x + b1) + b2

Expanding it:

y = (W2*W1)*x + (W2*b1 + b2)

This is still of the form:

y = A*x + c

So even if we stack many linear layers, the network can only learn linear
relationships.

Without activation functions, the neural network cannot learn complex
patterns like:

- Image classification
- Language understanding
- Non-linear decision boundaries
- Complex function approximation

Activation functions solve this problem.

After each linear transformation, an activation function applies a
non-linear operation element-wise.

This non-linearity gives neural networks the power to learn complex
patterns.

--------------------------------------------------
ReLU: The Simplest Nonlinearity
--------------------------------------------------

ReLU stands for Rectified Linear Unit.

It is defined as:

ReLU(x) = max(0, x)

The rule is simple:

If x > 0:
Output x

If x <= 0:
Output 0

Examples:

ReLU(3.5) = 3.5
ReLU(0) = 0
ReLU(-2.7) = 0
ReLU(100) = 100

The shape of ReLU looks like a hockey stick.

For negative inputs:
The graph is flat at 0.

For positive inputs:
The graph is a straight line with slope 1.

--------------------------------------------------
Why ReLU Became Dominant
--------------------------------------------------

Before ReLU, sigmoid and tanh were commonly used activation functions.

Sigmoid squashes values into the range:

(0, 1)

Tanh squashes values into the range:

(-1, 1)

They worked, but they had a major problem:

Vanishing gradient problem.

--------------------------------------------------
Vanishing Gradient Problem
--------------------------------------------------

Sigmoid and tanh saturate for large positive or negative inputs.

This means their output changes very little for large inputs.

In saturated regions, their derivative becomes almost zero.

During backpropagation, gradients are multiplied through many layers.

If each layer has a very small derivative, the gradient becomes smaller
and smaller.

This causes the gradient to shrink exponentially.

As a result:

- Earlier layers learn very slowly
- Deep layers may barely learn
- Training deep networks becomes difficult

--------------------------------------------------
How ReLU Helps
--------------------------------------------------

ReLU fixes this problem for positive inputs.

The derivative of ReLU is 1 for all x > 0.

This means gradients pass through unchanged for positive inputs.

Because of this:

- Gradient signals survive through many layers
- Deep networks train faster
- ReLU works better than sigmoid/tanh in many hidden layers

--------------------------------------------------
Practical Advantages of ReLU
--------------------------------------------------

1. Computationally cheap

ReLU only needs a comparison with zero.

It does not need exponentials or divisions.

2. Sparse activation

Many neurons output exactly 0.

This creates sparse representations.

Sparse representations can be more efficient and may generalize better.

3. Easy to implement

ReLU can be implemented in one line:

return max(0, x)

--------------------------------------------------
Gradient of ReLU
--------------------------------------------------

The derivative of ReLU is:

d/dx ReLU(x) = 1, if x > 0
d/dx ReLU(x) = 0, if x < 0

At x = 0:

ReLU has a sharp corner.

So it is technically not differentiable at x = 0.

In practice, frameworks usually define the derivative at x = 0 as 0.

Some may use 0.5 or 1.

This usually does not cause problems because hitting exactly x = 0 is rare
with floating-point numbers.

--------------------------------------------------
Why ReLU Gradient Is Useful
--------------------------------------------------

For positive inputs, ReLU has gradient 1.

This means the gradient does not shrink while passing through ReLU.

Compare this with sigmoid:

The maximum gradient of sigmoid is only 0.25 at x = 0.

After 10 sigmoid layers, the gradient may shrink by:

0.25^10 ≈ 10^(-6)

This is extremely small.

That is why sigmoid can cause vanishing gradients in deep networks.

ReLU avoids this problem in the positive region.

--------------------------------------------------
Dead Neuron Problem
--------------------------------------------------

The main weakness of ReLU is the dead neuron problem.

For negative inputs:

ReLU output = 0
ReLU gradient = 0

If a neuron's weighted input is negative for every training example, then:

- It always outputs 0
- It receives zero gradient
- Its weights stop updating
- It contributes nothing to the network

Such a neuron is called a dead neuron.

--------------------------------------------------
How Neurons Die
--------------------------------------------------

A neuron can die when:

1. A large gradient update changes the weights too much
2. The neuron's pre-activation becomes negative for all inputs
3. Since the input is always negative, ReLU output is always 0
4. Since the gradient is also 0, the weights never update again

The neuron becomes permanently inactive.

This can happen more often when the learning rate is too high.

In some networks, a significant number of neurons can die.

--------------------------------------------------
Solutions to Dead Neuron Problem
--------------------------------------------------

Some ReLU variants solve or reduce this issue.

1. Leaky ReLU

Leaky ReLU allows a small negative slope for negative inputs.

Instead of outputting 0 for negative x, it outputs:

alpha * x

So negative inputs still get a small gradient.

2. ELU

ELU uses an exponential curve for negative inputs.

This gives smoother behavior than ReLU.

3. GELU / Swish

GELU and Swish are smooth activation functions.

They avoid having exactly zero gradient in many regions.

4. Careful Initialization

Proper weight initialization reduces the chance of neurons starting in the
dead region.

Example:

He initialization is commonly used with ReLU.

--------------------------------------------------
Where ReLU Is Used
--------------------------------------------------

ReLU is the default activation function for many neural network
architectures.

It is commonly used in:

1. Convolutional Neural Networks

CNNs use ReLU in hidden layers.

AlexNet in 2012 was one of the first major successful CNNs that used ReLU.

After that, many CNN architectures used ReLU or its variants.

2. Fully Connected Layers

ReLU is commonly used in hidden layers of MLPs.

3. Generative Models

ReLU is used in generators and discriminators of GANs.

4. Residual Networks

ResNets often use ReLU after residual blocks.

--------------------------------------------------
Where ReLU Is Less Common
--------------------------------------------------

1. Output Layers

ReLU is usually not used in output layers when bounded outputs are needed.

For binary classification:
Use sigmoid.

For multi-class classification:
Use softmax.

2. Transformers

Transformers usually use GELU or Swish in feed-forward layers.

3. RNNs

RNNs often use tanh or sigmoid for recurrent connections.

Using ReLU in recurrent loops can sometimes cause exploding activations.

--------------------------------------------------
Summary
--------------------------------------------------

Activation functions add non-linearity to neural networks.

Without activation functions, multiple linear layers collapse into a single
linear function.

ReLU is defined as:

ReLU(x) = max(0, x)

ReLU outputs:

- x, if x > 0
- 0, if x <= 0

Advantages of ReLU:

- Simple
- Fast to compute
- Helps reduce vanishing gradients
- Produces sparse activations
- Works well in deep networks

Derivative of ReLU:

- 1 for x > 0
- 0 for x < 0

Main weakness:

Dead neuron problem.

A neuron can become dead if it always receives negative inputs and therefore
always outputs 0.

Solutions include:

- Leaky ReLU
- ELU
- GELU
- Swish
- Proper weight initialization

ReLU is mainly used in hidden layers of deep neural networks.
*/
'''