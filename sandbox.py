import numpy as np
import torch
import random
# action = np.random.normal(size=(10, 5, 4))
actions = []
for i in range(5) :
    action = []
    for j in range(4) :
        action.append(np.random.normal(size=3))
    actions.append(np.array(action))
actions = np.array(actions)
print(actions)
actions = np.transpose(actions, (1, 0, 2))
print(actions)
actions = torch.from_numpy(actions).float().transpose(0, 1)
print(actions)


# action = []
# for i in range(4) :
#     action.append(np.random.normal(size=4))
# action = np.array(action)
# print(action)
# print()
# print(np.transpose(action))


policy = np.array([0.1, 0.2, 0.3, 0.1])
action = []
for k in range(10000) :
    action.append(random.choices([i for i in range(4)], weights=policy)[0])
print(action)
print(len(list(filter(lambda x: x==0, action))))
print(len(list(filter(lambda x: x==1, action))))
print(len(list(filter(lambda x: x==2, action))))
print(len(list(filter(lambda x: x==3, action))))