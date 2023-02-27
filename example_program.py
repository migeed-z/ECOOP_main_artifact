import torch
from torch.fx.experimental.rewriter import RewritingTracer

print(torch.rand(1,2,0))

#
# class MyCode(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = torch.nn.Conv2d(
#             in_channels=512, out_channels=512, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x.dim() == 4
#         return x
#
#
# ast_rewriter = RewritingTracer()
# graph = ast_rewriter.trace(MyCode())
# print(graph.print_tabular())

# print(torch.bmm(torch.rand(1,8,3), torch.rand(1, 3, 2)).size())