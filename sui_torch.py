import numpy as np


class Tensor:
    def __init__(self, value, back_op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.back_op = back_op

    def __str__(self):
        str_val = str(self.value)
        str_val = '\t' + '\n\t'.join(str_val.split('\n'))
        str_bwd = str(self.back_op.__class__.__name__)
        return 'Tensor(\n' + str_val + '\n\tbwd: ' + str_bwd + '\n)'

    @property
    def shape(self):
        return self.value.shape

    def backward(self, deltas=None):
        if deltas is not None:
            # Check if the deltas shape matches the current tensor shape
            assert deltas.shape == self.value.shape, f'Expected gradient with shape {self.value.shape}, got {deltas.shape}'
            self.grad = deltas  # Add the incoming gradient to the current gradient
        else:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(f'Can only backpropagate a scalar, got shape {self.shape}')
            
            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')
            
            self.grad = 1.0

        # Recursively call the backpropagation operation to propagate the error back
        # Recursively call the backpropagation operation of the previous step
        if self.back_op is not None:
            self.back_op(self.grad)


def sui_sum(tensor):
    """
    Sums all the elements of a tensor to produce a scalar result.

    Formula: S = sum(A)
    Gradients:
    dS/dA = 1 (for each element in A)

    Each element in the tensor contributes equally to the sum, so the gradient
    for each element is the same (grad_output passed from backpropagation).
    """
    result = np.sum(tensor.value)

    def back_op(_):
        tensor.grad = np.ones_like(tensor.value)    # Every element contributes equally
        tensor.backward(tensor.grad)

    return Tensor(result, back_op)


def add(tensor_A, tensor_B):
    """
    Addition of two tensors A and B: C = A + B.

    Formula: C = A + B
    Gradients with respect to A and B:
    dC/dA = 1, dC/dB = 1

    Both tensors contribute equally to the result, so their gradients are identical.
    """
    result = tensor_A.value + tensor_B.value

    def back_op(grad_output):
        tensor_A.grad += grad_output  # Gradient with respect to A
        tensor_B.grad += grad_output  # Gradient with respect to B
        tensor_A.backward(tensor_A.grad)
        tensor_B.backward(tensor_B.grad)

    return Tensor(result, back_op)


def subtract(tensor_A, tensor_B):
    """
    Subtraction of two tensors A and B: C = A - B.

    Formula: C = A - B
    Gradients with respect to A and B:
    dC/dA = 1, dC/dB = -1

    The gradient for B is negative as it is subtracted from A.
    """
    result = tensor_A.value - tensor_B.value  # Element-wise subtraction.

    def back_op(grad_output):
        # Gradient with respect to A is the same as the grad_output.
        tensor_A.grad += grad_output
        # Gradient with respect to B is the negative of grad_output.
        tensor_B.grad -= grad_output
        tensor_A.backward(tensor_A.grad)
        tensor_B.backward(tensor_B.grad)

    return Tensor(result, back_op)


def multiply(tensor_A, tensor_B):
    """
    Element-wise multiplication of two tensors A and B: C = A * B.

    Formula: C = A * B
    Gradients with respect to A and B:
    dC/dA = B
    dC/dB = A

    Each element in A is multiplied by the corresponding element in B.
    """
    result = tensor_A.value * tensor_B.value  # Element-wise multiplication.

    def back_op(grad_output):
        # Gradient with respect to A is B element-wise multiplied by grad_output.
        tensor_A.grad+= tensor_B.value * grad_output
        # Gradient with respect to B is A element-wise multiplied by grad_output.
        tensor_B.grad+= tensor_A.value * grad_output
        tensor_A.backward(tensor_A.grad)
        tensor_B.backward(tensor_B.grad)

    return Tensor(result, back_op)


def relu(tensor):
    """
    Applies the ReLU activation function to a tensor: C = ReLU(A) = max(0, A).

    Formula: C = max(0, A)
    Gradients:
    dC/dA = 1, if A > 0
    dC/dA = 0, if A <= 0

    The gradient is 1 for positive elements and 0 for non-positive elements.
    """
    result = np.maximum(0, tensor.value)  # Element-wise ReLU application.

    def back_op(grad_output):
        # Gradient is 1 where tensor value is positive, otherwise 0.
        tensor.grad += grad_output * (tensor.value > 0)
        tensor.backward(tensor.grad)

    return Tensor(result, back_op)


def dot_product(tensor_A, tensor_B):
    """
    Matrix multiplication of two tensors (matrices) A and B: C = A @ B.

    Formula: C = A * B (matrix multiplication)
    Gradients with respect to A and B:
    dC/dA = B^T (transpose of B)
    dC/dB = A^T (transpose of A)

    The gradients are the transposed versions of the input matrices.
    """
    result = np.dot(tensor_A.value, tensor_B.value)  # Matrix multiplication (dot product).

    def back_op(grad_output):
        # Gradient with respect to A is the transpose of B multiplied by grad_output.
        tensor_A.grad += np.dot(grad_output, tensor_B.value.T)
        # Gradient with respect to B is the transpose of A multiplied by grad_output.
        tensor_B.grad += np.dot(tensor_A.value.T, grad_output)
        tensor_A.backward(tensor_A.grad)
        tensor_B.backward(tensor_B.grad)

    return Tensor(result, back_op)
