# Owner(s): ["module: fx"]
import operator
import unittest
from torch.fx import GraphModule, symbolic_trace
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import transform_all_constraints
from torch.fx.experimental.migrate_gradual_types.z3_types import tensor_type, D, z3_dyn
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.tensor_type import Dyn, TensorType
import torch
from timeit import default_timer as timer
from datetime import timedelta

try:
    import z3  # type: ignore[import]
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

try:
    from torchvision import models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

class ConvTest(unittest.TestCase):

    def test_conv_intro_example(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=2,
                                             kernel_size=2, stride=2,
                                             padding=2, groups=2, bias=False, dilation=2)

                self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=2,
                                             kernel_size=2, stride=2,
                                             padding=2, groups=2, bias=False, dilation=2)


            def forward(self, x: Dyn):
                self.conv1(x)
                return self.conv2(x)


        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        start = timer()

        # first, check rank properties
        transformed = transform_all_constraints(traced)
        solver1 = z3.Solver()
        solver1.add(transformed)
        self.assertEqual(solver1.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        solver1.add(x == tensor_type.tensor1(d1) or
                    x == tensor_type.tensor2(d1, d2) or
                    x == tensor_type.tensor3(d1, d2, d3))

        # Here, we prove that input cannot be rank1, rank2 or rank3
        assert solver1.check() == z3.unsat
        print('we proved that the input cannot be rank-1, rank-2 or rank-3')

        # Here we prove the input must be rank-4
        transformed = transform_all_constraints(traced)
        solver3 = z3.Solver()
        solver3.add(transformed)
        x = z3.Const(1, tensor_type)
        solver3.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        assert solver3.check() == z3.sat
        print('we proved that the input must be rank-4')


        #Finally here, we prove that it cannot be fully static by checking every dimension
        solver3.add(s11 != 0)
        assert solver3.check() == z3.sat
        print('we proved that the first dimension is a number')

        solver3.add(s22 != 0)
        assert solver3.check() == z3.unsat
        print('we proved that the second dimension cannot be a number')

        end = timer()
        print(timedelta(seconds=end-start))

    def test_conv_right_example_w_arithmetic(self):
        # we fix the example as we suggested in the introduction
        # by removing the extra convolution, then we impose arithmetic constraints
        # on the program

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

                self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=2,
                                             kernel_size=2, stride=2,
                                             padding=2, groups=2, bias=False, dilation=2)
            def forward(self, x: Dyn):
                return self.conv2(x)


        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        start = timer()


        # here we prove the input must be rank-4
        transformed = transform_all_constraints(traced)

        solver3 = z3.Solver()
        solver3.add(transformed)
        x = z3.Const(1, tensor_type)
        print('We added a constraint that the input must be rank-4')
        solver3.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        assert solver3.check() == z3.sat

        # finally here, we prove that it cannot be fully static by checking every dimension
        print('We added four constraints that all dimensions must be numbers')
        solver3.add(s11 != 0)
        solver3.add(s22 != 0)
        solver3.add(s33 != 0)
        solver3.add(s44 != 0)

        assert solver3.check() == z3.sat
        print('We proved that all dimensions must be numbers')

        print('We added arithmetic constraints')
        solver3.add(s1 > 5)
        solver3.add(s1 < 20)
        solver3.add(s3 > 5)
        solver3.add(s3 < 20)
        solver3.add(s4 > 2)
        solver3.add(s4 < 10)

        assert solver3.check() == z3.sat
        print('our constraints are satisfiable with the following input annotation: ')
        print(f'TensorType({solver3.model()[x].arg(0).arg(1)}, {solver3.model()[x].arg(1).arg(1)},'
              f' {solver3.model()[x].arg(2).arg(1)}, {solver3.model()[x].arg(3).arg(1)})')
        end = timer()
        print(timedelta(seconds=end-start))


class BmmTest(unittest.TestCase):
    def test_bmm(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()

            def forward(self, x: TensorType([1, 3, 2]), y: TensorType([1, 3, 2])):
                return torch.bmm(x, y)

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())

        start = timer()
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.unsat)
        print('Migration space is empty. Program is ill-typed')
        end = timer()
        print(timedelta(seconds=end-start))


@skipIfNoTorchVision
class TestAlexNet(unittest.TestCase):
    def test_alexnet3(self):
        alexnet = models.alexnet()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(alexnet)
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                n.type = Dyn

        input = z3.Const(1, tensor_type)
        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        start = timer()
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver0 = z3.Solver()
        solver0.add(constraints)
        self.assertEqual(solver0.check(), z3.sat)
        solver0.add(input == tensor_type.tensor1(d1) or
                    input == tensor_type.tensor2(d1, d2) or
                    input == tensor_type.tensor3(d1, d2, d3))
        assert solver0.check() == z3.unsat
        print('We proved that the input cannot be rank-1, rank-2 or rank-3')


        solver = z3.Solver()
        solver.add(constraints)
        v1, v2, v3, v4 = z3.Ints('v1 v2 v3 v4')
        solver.add(input == tensor_type.tensor4(D(1, v1), D(1, v2), D(1, v3), D(1, v4)))
        assert solver.check() == z3.sat
        print('We proved that the input is rank-4')
        end = timer()
        print('This is the overall time it took to compute the results so far:')
        print(timedelta(seconds=end-start))

        print('We added an arithmetic constraint:')
        solver.add(v1 > 5)
        assert solver.check() == z3.sat
        print('We added an arithmetic constraint:')
        print('our constraints are satisfiable with the following input annotation: ')
        print(f'TensorType({solver.model()[input].arg(0).arg(1)}, {solver.model()[input].arg(1).arg(1)},'
              f' {solver.model()[input].arg(2).arg(1)}, {solver.model()[input].arg(3).arg(1)})')

        end = timer()
        print('This is the time it took to compute this result')
        print(timedelta(seconds=end-start))

@skipIfNoTorchVision
class TestResNet(unittest.TestCase):
    def test_resnet502(self):
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        s1, s2, s3, s4 = z3.Ints('s1 s2 s3 s4')
        s11, s22, s33, s44 = z3.Ints('s11 s22 s33 s44')
        d1, d2, d3, d4 = D(s11, s1), D(s22, s2), D(s33, s3), D(s44, s4),

        start = timer()
        constraints = transform_all_constraints(traced, counter=0)
        input = z3.Const(1, tensor_type)

        # input is not rank 1, 2 or 3
        solver0 = z3.Solver()
        solver0.add(constraints)
        self.assertEqual(solver0.check(), z3.sat)
        solver0.add(input == tensor_type.tensor1(d1) or
                    input == tensor_type.tensor2(d1, d2) or
                    input == tensor_type.tensor3(d1, d2, d3))
        assert solver0.check() == z3.unsat
        print('We proved that the input cannot be rank-1, rank-2 or rank-3')

        # since it is rank-4, check if it can be fully static
        solver = z3.Solver()
        solver.add(constraints)
        solver.add(input == tensor_type.tensor4(D(1, s1), D(1, s2), D(1, s3), D(1, s4)))
        assert solver.check() == z3.sat
        end = timer()
        print('This is the overall time it took to compute the results so far:')
        print(timedelta(seconds=end-start))
        print('We proved that the input is rank-4')

        # then add arithmetic constraints. We annotate all but one input.
        solver.add(input == tensor_type.tensor4(D(1, s1), D(1, 3), D(1, 224), D(1, 224)))
        solver.add(s1 > 4)
        print('We added an arithmetic constraints on the first dimension and fix the other three dimensions')


        assert solver.check() == z3.sat
        print('our constraints are satisfiable with the following input annotation: ')
        print(f'TensorType({solver.model()[input].arg(0).arg(1)}, {solver.model()[input].arg(1).arg(1)},'
          f' {solver.model()[input].arg(2).arg(1)}, {solver.model()[input].arg(3).arg(1)})')

        end = timer()
        print('This is overall time it took to compute the result after adding arithmetic constraints:')
        print(timedelta(seconds=end-start))

if __name__ == '__main__':
    unittest.main()

