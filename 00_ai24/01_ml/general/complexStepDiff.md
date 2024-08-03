# Complex Step Differentiation

* a numerical differentiation technique used to approximate the derivative of a real-valued function with respect to one of its variables.
* particularly appealing due to its simplicity and high accuracy, making it a useful tool in optimization,


### the trick

* exploit the properties of complex numbers to bypass the subtraction operation found in finite difference methods, which can introduce significant numerical errors due to the limitations of floating-point arithmetic
* using a complex perturbation, we can effectively avoid the catastrophic cancellation that plagues traditional methods, especially for very small step sizes.

![alt text](image-1.png)