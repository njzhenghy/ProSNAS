
import torch
import torch.nn.functional as F
def _projection_unit_simplex(x1):
  """Projection onto the unit simplex."""
  x = torch.reshape(x1,[-1])
  s = 1.0
  n_features = x.shape[0]
  u = torch.sort(x,0,descending=True)[0]
  cumsum_u = torch.cumsum(u, dim=0)
  ind = (torch.arange(n_features)+1).to(x.device)
  ind = torch.reshape(ind, cumsum_u.shape)
  cond = s / ind + (u - cumsum_u / ind) > 0
  idx = torch.count_nonzero(cond)
  tmp = torch.clip(s / idx + (x - cumsum_u[idx - 1] / idx), min=0)
  return torch.reshape(tmp, x1.shape)


def projection_simplex(x, value: float = 1.0):
  r"""Projection onto a simplex:

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    p \ge 0, p^\top 1 = \text{value}

  By default, the projection is onto the probability simplex.

  Args:
    x: vector to project, an array of shape (n,).
    value: value p should sum to (default: 1.0).
  Returns:
    projected vector, an array with the same shape as ``x``.
  """
  if value is None:
    value = 1.0
  return value * _projection_unit_simplex(x / value)

def projectionA(A):
  A_pro = torch.zeros_like(A)
  for i in range(len(A)):
    A_pro[i,:]= projection_simplex(A[i,:])
  return A_pro

  
def projectionB(B):
  B_pro = torch.rand_like(B)
  B_pro[:, 3, 1] = -1e8
  B_pro = F.softmax(B_pro, dim=-1)
  for layer in range(len(B)):
    if layer == 0:
      B_pro[layer][0] = projection_simplex(B[layer][0])
    elif layer == 1:
      B_pro[layer][0] = projection_simplex(B[layer][0])
      B_pro[layer][1] = projection_simplex(B[layer][1])

    elif layer == 2:
      B_pro[layer][0] = projection_simplex(B[layer][0])
      B_pro[layer][1] = projection_simplex(B[layer][1])
      B_pro[layer][2] = projection_simplex(B[layer][2])
    else:
      B_pro[layer][0] = projection_simplex(B[layer][0])
      B_pro[layer][1] = projection_simplex(B[layer][1])
      B_pro[layer][2] = projection_simplex(B[layer][2])
      B_pro[layer][3][0] = 1.
      B_pro[layer][3][1] = 0.
  return B_pro
# def projectionB(B):
#   B_pro = torch.rand_like(B)
#   B_pro[:, 0, 0] = -1e8
#   B_pro[:, 3, 2] = -1e8
#   B_pro = F.softmax(B_pro, dim=-1)
#   for layer in range(len(B)):
#     if layer == 0:
#       B_pro[layer][0][1:] = projection_simplex(B[layer][0][1:])
#     elif layer == 1:
#       B_pro[layer][0][1:] = projection_simplex(B[layer][0][1:])
#       B_pro[layer][1] = projection_simplex(B[layer][1])

#     elif layer == 2:
#       B_pro[layer][0][1:] = projection_simplex(B[layer][0][1:])
#       B_pro[layer][1] = projection_simplex(B[layer][1])
#       B_pro[layer][2] = projection_simplex(B[layer][2])
#     else:
#       B_pro[layer][0][1:] = projection_simplex(B[layer][0][1:])
#       B_pro[layer][1] = projection_simplex(B[layer][1])
#       B_pro[layer][2] = projection_simplex(B[layer][2])
#       B_pro[layer][3][:2] = projection_simplex(B[layer][3][:2])
#   return B_pro

# A = torch.randn((9,6))
# A_pro = projectionA(A)
#
# B = torch.randn(8,4,3)
# B_pro = projectionB(B)
# pass