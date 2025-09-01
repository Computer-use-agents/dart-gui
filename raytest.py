import ray

# 直接本地 init
ray.init()

@ray.remote
def f(x):
    return x * x

print(ray.get([f.remote(i) for i in range(5)]))
