import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.array(x)
    return np.where(x > 0, x, alpha * x)
    pass

'''
NOTES - 
Dead Neuron Problem and Leaky ReLU Notes

--------------------------------------------------
The Dead Neuron Problem
--------------------------------------------------

ReLU outputs exactly 0 for any negative input.

ReLU is defined as:

ReLU(x) = max(0, x)

This means:

If x > 0:
Output = x

If x <= 0:
Output = 0

The problem happens when a neuron's pre-activation is negative for every
training example.

In that case:

- The output of the neuron is always 0
- The gradient of the neuron is always 0
- The weights of the neuron never update
- The neuron contributes nothing to the network

Such a neuron is called a dead neuron.

Once a neuron becomes dead, it usually cannot recover because the gradient
is exactly 0.

--------------------------------------------------
Why Dead Neurons Happen
--------------------------------------------------

Dead neurons can happen more often than expected.

Common reasons:

1. Large gradient updates

A large gradient update can change the weights or bias too much.

This may push the neuron's pre-activation into the negative region for all
training examples.

2. High learning rate

If the learning rate is too high, the model updates weights aggressively.

This can cause many neurons to become dead in the first few training steps.

3. Zero gradient for negative inputs

Once the neuron always receives negative inputs, ReLU always outputs 0.

Since the gradient is also 0, the neuron cannot update its weights anymore.

--------------------------------------------------
Leaky ReLU: A Simple Fix
--------------------------------------------------

Leaky ReLU is a modified version of ReLU.

Instead of making the output exactly 0 for negative inputs, it allows a
small negative slope.

Leaky ReLU is defined as:

LeakyReLU(x) = x,       if x >= 0
LeakyReLU(x) = alpha*x, if x < 0

Here, alpha is a small positive constant.

Usually:

alpha = 0.01

For positive inputs, Leaky ReLU behaves exactly like ReLU.

For negative inputs, it multiplies the input by alpha instead of making it
0.

This small negative slope is called the leak.

The leak allows a small amount of signal and gradient to pass through even
for negative inputs.

--------------------------------------------------
Examples of Leaky ReLU
--------------------------------------------------

Let alpha = 0.01.

LeakyReLU(3.0) = 3.0

LeakyReLU(0) = 0

LeakyReLU(-2.0) = 0.01 * (-2.0)
                = -0.02

LeakyReLU(-100) = 0.01 * (-100)
                 = -1.0

So unlike ReLU, Leaky ReLU does not completely block negative values.

It allows a small negative output.

--------------------------------------------------
The Role of Alpha
--------------------------------------------------

The parameter alpha controls the slope for negative inputs.

Different values of alpha give different behavior.

1. alpha = 0

This becomes standard ReLU.

There is no leak.

Negative inputs become 0.

2. alpha = 0.01

This is the most common default value.

Negative inputs are scaled down by 100 times.

3. alpha = 0.1 or 0.2

This gives a larger leak.

It allows more gradient flow for negative inputs.

4. alpha = 1

The function becomes:

f(x) = x

This means the function becomes identity.

There is no non-linearity.

5. alpha > 1

The negative side has a steeper slope than the positive side.

This is unusual but mathematically valid.

--------------------------------------------------
Key Insight
--------------------------------------------------

The main idea is:

As long as alpha is not 0, the neuron cannot completely die.

Even when the pre-activation is negative, the gradient is still nonzero.

For negative inputs, the gradient is alpha.

So the weights can still update during backpropagation.

This helps prevent the dead neuron problem.

--------------------------------------------------
Gradient of Leaky ReLU
--------------------------------------------------

The derivative of Leaky ReLU is:

d/dx LeakyReLU(x) = 1,     if x >= 0
d/dx LeakyReLU(x) = alpha, if x < 0

Compare this with ReLU:

d/dx ReLU(x) = 1, if x > 0
d/dx ReLU(x) = 0, if x < 0

For ReLU, the derivative for negative inputs is exactly 0.

For Leaky ReLU, the derivative for negative inputs is alpha.

Alpha is small, but it is not zero.

--------------------------------------------------
Backpropagation with Leaky ReLU
--------------------------------------------------

During backpropagation:

1. Positive region

If x >= 0:

Gradient is multiplied by 1.

So the gradient passes through at full strength.

2. Negative region

If x < 0:

Gradient is multiplied by alpha.

For example, if alpha = 0.01, the gradient becomes 1% of its original
value.

The gradient becomes smaller, but it still flows.

This means every neuron can still receive some gradient, even if its input
is negative.

--------------------------------------------------
Parametric ReLU
--------------------------------------------------

Parametric ReLU is also called PReLU.

It is an extension of Leaky ReLU.

In normal Leaky ReLU, alpha is fixed manually.

Example:

alpha = 0.01

In PReLU, alpha is not fixed.

Instead, alpha is learned during training, just like weights and biases.

PReLU is defined as:

PReLU(x) = x,       if x >= 0
PReLU(x) = alpha*x, if x < 0

The formula is the same as Leaky ReLU.

The difference is that alpha is trainable.

--------------------------------------------------
Behavior of PReLU
--------------------------------------------------

If the learned alpha becomes close to 0:

PReLU behaves like ReLU.

If the learned alpha becomes close to 1:

PReLU behaves almost like a linear function.

In practice, learned alpha values often become somewhere between:

0.1 and 0.3

This allows the network to choose the best negative slope automatically.

--------------------------------------------------
Where Leaky ReLU Is Used
--------------------------------------------------

Leaky ReLU is used in many deep learning models.

1. GANs

GAN stands for Generative Adversarial Network.

Leaky ReLU is commonly used in discriminator networks.

In DCGAN, alpha = 0.2 was recommended for the discriminator.

2. Object Detection

Some object detection models, such as YOLO, use Leaky ReLU.

3. Deep Networks with Dead Neurons

If many neurons are producing zero output after training, switching from
ReLU to Leaky ReLU can help.

4. Diagnostic Use

If Leaky ReLU performs much better than ReLU on a task, it may indicate
that dead neurons were hurting performance.

--------------------------------------------------
Summary
--------------------------------------------------

ReLU is defined as:

ReLU(x) = max(0, x)

For negative inputs, ReLU outputs 0 and has gradient 0.

This can cause the dead neuron problem.

A dead neuron:

- Always outputs 0
- Receives zero gradient
- Stops updating
- Contributes nothing to the network

Leaky ReLU fixes this by allowing a small slope for negative inputs.

Leaky ReLU is defined as:

LeakyReLU(x) = x,       if x >= 0
LeakyReLU(x) = alpha*x, if x < 0

For negative inputs, the gradient is alpha instead of 0.

This allows gradients to keep flowing and helps prevent neurons from dying.

Common alpha value:

alpha = 0.01

PReLU is a learnable version of Leaky ReLU where alpha is learned during
training.

Leaky ReLU is commonly used in:

- GAN discriminators
- Object detection models
- Deep networks where dead neurons are a problem

'''