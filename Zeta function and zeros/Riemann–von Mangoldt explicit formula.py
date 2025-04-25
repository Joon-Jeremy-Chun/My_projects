# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:33:04 2025

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath

# 비자명 영점 (t값), 실제로 더 많은 영점을 이용 가능
# riemann_zeros_t = [
#     14.134725, 21.022040, 25.010856, 30.424876, 32.935062,
#     37.586178, 40.918719, 43.327073, 48.005151, 49.773832
# ]

riemann_zeros_t = [
    14.1347251417347,
    21.022039638771554,
    25.01085758014569,
    30.4248761258595,
    32.9350615877392,
    37.5861781588257,
    40.9187190121475,
    43.327073280914999,
    48.005150881167159,
    49.773832477672302,
    52.97032147771452,
    56.446247697063394,
    59.347044002602,
    60.831778524609809,
    65.112544048081,
    67.079810529494292,
    69.546401711173,
    72.067157674481845,
    75.7046906990839,
    77.144840068874,
    79.337375020249928,
    82.910380854086,
    84.735492980517,
    87.425274613337,
    88.809111207634,
    92.491899270558,
    94.651344040519,
    95.870634228245,
    98.831194218193,
    101.31785100572
]

# riemann_zeros_t = [
#     3,   #look at here
#     21.022039638771554,
#     25.01085758014569,
#     30.4248761258595,
#     32.9350615877392,
#     37.5861781588257,
#     40.9187190121475,
#     43.327073280914999,
#     48.005150881167159,
#     49.773832477672302,
#     52.97032147771452,
#     56.446247697063394,
#     59.347044002602,
#     60.831778524609809,
#     65.112544048081,
#     67.079810529494292,
#     69.546401711173,
#     72.067157674481845,
#     75.7046906990839,
#     77.144840068874,
#     79.337375020249928,
#     82.910380854086,
#     84.735492980517,
#     87.425274613337,
#     88.809111207634,
#     92.491899270558,
#     94.651344040519,
#     95.870634228245,
#     98.831194218193,
#     101.31785100572
# ]

n_zeros = len(riemann_zeros_t)
label_text = f"Explicit formula approx. ({n_zeros} zeros)"
# ψ(x) 명시적 공식으로 표현
def explicit_psi(x, zeros_t):
    sum_term = 0.0
    for t in zeros_t:
        rho = 0.5 + 1j*t
        term = (x ** rho) / rho + (x ** (1 - rho)) / (1 - rho)
        sum_term += term.real
    return x - sum_term - np.log(2*np.pi) - 0.5*np.log(1 - x**(-2))

# 실제 ψ(x) (정의에 의한 계산)
def psi(x, primes):
    val = 0.0
    for p in primes:
        k = 1
        while p**k <= x:
            val += np.log(p)
            k += 1
    return val

# 소수 구하기 (에라토스테네스 체)
def prime_sieve(n):
    sieve = np.ones(n+1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return np.nonzero(sieve)[0]

# 메인 코드
x_values = np.linspace(2, 100, 500)
primes = prime_sieve(100)
psi_actual = [psi(x, primes) for x in x_values]
psi_approx = [explicit_psi(x, riemann_zeros_t) for x in x_values]

plt.figure(figsize=(10,6))
plt.step(x_values, psi_actual, 'r', label="ψ(x), actual", where='post')
plt.plot(x_values, psi_approx, 'b-', label=label_text)

plt.xlabel('x')
plt.ylabel('ψ(x)')
plt.legend()
plt.grid()
plt.title('Explicit Formula using Riemann Zeros for ψ(x)')
plt.show()
