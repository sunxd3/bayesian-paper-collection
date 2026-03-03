import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import numpy as np

class SolOp(Op):
    def __init__(self, sol_op_jax_jitted):
        self.sol_op_jax_jitted = sol_op_jax_jitted

    def make_node(self, *inputs):
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        outputs = [pt.dmatrix()]
        return Apply(self, inputs, outputs)
    
    def perform(self, node, inputs, outputs):
        result = self.sol_op_jax_jitted(*inputs)
        outputs[0][0] = np.asarray(result, dtype="float64")
    
    def grad(self, inputs, output_grads):
        (gz,) = output_grads
        return vjp_sol_op(inputs, gz)
    
class VJPSolOp(Op):
    def __init__(self, vjp_sol_op_jax_jitted):
        self.vjp_sol_op_jax_jitted = vjp_sol_op_jax_jitted 

    def make_node(self, *inputs, output_grads):
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        inputs += [pt.as_tensor_variable(output_grads)]
        outputs = [inputs[i].type() for i in range(len(inputs)-1)]
        return Apply(self, inputs, outputs)
    
    def perform(self, node, inputs, outputs):
        *inputs, output_grads = inputs
        result = self.vjp_sol_op_jax_jitted(*inputs, output_grads)
        for i in range(len(result)):
            outputs[i][0] = np.asarray(result[i], dtype="float64")