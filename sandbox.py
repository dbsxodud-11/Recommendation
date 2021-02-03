import numpy as np
from scipy import sparse
import torch
import random
# # action = np.random.normal(size=(10, 5, 4))
# actions = []
# for i in range(5) :
#     action = []
#     for j in range(4) :
#         action.append(np.random.normal(size=3))
#     actions.append(np.array(action))
# actions = np.array(actions)
# print(actions)
# actions = np.transpose(actions, (1, 0, 2))
# print(actions)
# actions = torch.from_numpy(actions).float().transpose(0, 1)
# print(actions)


# # action = []
# # for i in range(4) :
# #     action.append(np.random.normal(size=4))
# # action = np.array(action)
# # print(action)
# # print()
# # print(np.transpose(action))


# policy = np.array([0.1, 0.2, 0.3, 0.1])
# action = []
# for k in range(10000) :
#     action.append(random.choices([i for i in range(4)], weights=policy)[0])
# print(action)
# print(len(list(filter(lambda x: x==0, action))))
# print(len(list(filter(lambda x: x==1, action))))
# print(len(list(filter(lambda x: x==2, action))))
# print(len(list(filter(lambda x: x==3, action))))

x = np.array([[1, 0, 0],[5, 2, 0],[0, 0, 3],[0, 0, 1]])
x = sparse.csr_matrix(x)
row = x
col = x.transpose()
# print(row)
# print()
# print(col)
# print(col.tocsr())

x1 = np.random.normal(size=(3, 4))
# y = np.repeat(np.expand_dims(x1, axis=0), 3, axis=0)
# print(x1)
# print(y)
# print(x1)
# print(row.indices)
# print(row.shape)
# print(col.indices)
# print(col.shape)
# lookup = x1[row.indices]
# print(lookup)
# print(x1[col.indices])
# expanded_lookup = np.expand_dims(lookup, axis=1)
# print(expanded_lookup)
# print(np.transpose(expanded_lookup, axes=[0,2,1]))
# print(np.matmul(np.transpose(expanded_lookup, axes=[0,2,1]), expanded_lookup))


a = np.array([[0,1,0,0],[0,0,1,1],[0,0,1,0]])
a = torch.tensor(a)
print(a.view(3, 1, 4))
# a = sparse.csr_matrix(a)
# print(a.indices)
# print(a.indptr)

a = np.array([[1],[3],[5]])
print(np.sum(a, axis=1))