import torch

# 1. Create a tensor from a list
data = [[1., 2.], [3., 4.]]
t_data = torch.tensor(data)
print(t_data)

# 2. Create a tensor of zeros
shape = (2, 3)
r_zeros = torch.zeros(shape)
print(r_zeros)

# 3. Create a tensor of random values
r_random = torch.randn(shape)
print(r_random)

# 4. Create a tensor of random values with the same shape as r_zeros
template = torch.randn_like(r_zeros)
print(template)

print(template.shape)
print(template.dtype)
print(template.device)

## 1. require_grad
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x)
print(x.requires_grad)

### Operations
## 1. Element-wise operations
ele_mat = t_data * torch.tensor([[2., 0.], [0., 2.]])
print(f"ele_mat: {ele_mat}")

## 2. Matrix multiplication
mat_mul = t_data @ torch.tensor([[2., 3.], [4., 5.]])
print(f"mat_mul: {mat_mul}")

### Reduction operations
## 1. Sum
sum_all = t_data.sum()
print(f"sum_all: {sum_all}")

## 2. Mean
mean_all = t_data.mean()
print(f"mean_all: {mean_all}")

mean_0 = t_data.mean(dim=0)
print(f"mean_0: {mean_0}")

mean_1 = t_data.mean(dim=1)
print(f"mean_1: {mean_1}")

### Indexing and slicing
## 1. Indexing
ind_tensor = torch.tensor([[10., 20., 30.], [40., 50., 60.], [70., 80., 90.], [100., 110., 120.]])
first_row = ind_tensor[0]
print(f"first_row: {first_row}")
col_2_row_3 = ind_tensor[2, 1]
print(f"col_2_row_3: {col_2_row_3}")

## 2. argmax
argmax_0 = ind_tensor.argmax(dim=0)
print(f"argmax_0: {argmax_0}")