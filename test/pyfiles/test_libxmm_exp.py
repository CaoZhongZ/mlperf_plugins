import numpy as np

log2_e = 1.442695
half = 0.5
c2 = 0.240226507
c1 = 0.452920674
c0 = 0.713483036

def polynomial(y):
    c2 = 0.240226507
    c1 = 0.452920674
    c0 = 0.713483036
    t1 = y * c2 + c1
    return y * t1 + c0

def libxmm_exp(x):
    x1 = x * log2_e + half
    y = x1 - np.round(x1)
    two_to_y = polynomial(y)
    r = np.floor(x1)
    exp = two_to_y * (2**r)
    return exp

def ibert_exp(x):
    c = [0.35815147, 0.96963238, 1.0]
    r = np.floor(x / -log2_e)
    p = x + r * np.log(2)
    exp =  (p * c[0] + c[1]) * p + c[2]
    return exp * (2**(-r))

x = np.random.randn(1024, 1024)
mask = x < 0
x = x * mask + x * (mask - 1)

libxmm_exp_x = libxmm_exp(x)
real_exp_x = np.exp(x)
ibert_exp_x = ibert_exp(x)

print('----------------libxmm_average_error-----------------------')
print(np.mean(real_exp_x - libxmm_exp_x))
print('----------------ibert_average_error-----------------------')
print(np.mean(real_exp_x - ibert_exp_x))
print('----------------libxmm_max_error-----------------------')
print(np.max(np.abs(real_exp_x - libxmm_exp_x)))
print('----------------ibert_max_error-----------------------')
print(np.max(np.abs(real_exp_x - ibert_exp_x)))

y = np.random.randn(1024, 1024)
y = np.clip(y, -0.5, 0.5)
poly_y = polynomial(y)
real_2_y = 2**y
print('----------------poly_average_error-----------------------')
print(np.mean(real_2_y - poly_y))
