# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 22:40:57 2025

@author: joonc
"""

# %% [1] Imports
import mpmath as mp
mp.mp.dps = 8   # precision

# Helper function for sqrt safely
def S(x):
    return mp.sqrt(x)
# %% [2] Integral #1 : Order = dx dy dz
# 
# V = 2 ∫_{z=0..1} [ ∫_{y=z..1} ∫_{x=-√(1-y²)..√(1-y²)} dx dy
#                 + ∫_{y=0..z} ( ∫_{x=-√(1-y²)..-√(z²-y²)} dx
#                               + ∫_{x=√(z²-y²)..√(1-y²)} dx ) dy ] dz

def integrand1(x, y):
    return 1.0  # we're integrating 1 dV

def inner_y_ge_z(y):
    return 2 * S(1 - y**2)   # integral over x

def inner_y_le_z(y, z):
    if y == 0 and z == 0:
        return 0
    left_len  = -S(z**2 - y**2) - (-S(1 - y**2))
    right_len = S(1 - y**2) - S(z**2 - y**2)
    return left_len + right_len

def inner_integral1(z):
    # y from z to 1
    part1 = mp.quad(lambda yy: inner_y_ge_z(yy), [z, 1])
    # y from 0 to z
    part2 = mp.quad(lambda yy: inner_y_le_z(yy, z), [0, z])
    return part1 + part2

V1 = 2 * mp.quad(lambda zz: inner_integral1(zz), [0, 1])
print("Integral 1 (dx dy dz) =", V1)
# %% [3] Integral #2 : Order = dx dz dy
#
# V = 2 ∫_{y=0..1} [ ∫_{z=0..y} ∫_{x=-√(1-y²)..√(1-y²)} dx dz
#                 + ∫_{z=y..1} ( ∫_{x=-√(1-y²)..-√(z²-y²)} dx
#                               + ∫_{x=√(z²-y²)..√(1-y²)} dx ) dz ] dy

def inner_z_le_y(z, y):
    return 2 * S(1 - y**2)

def inner_z_ge_y(z, y):
    left_len  = -S(z**2 - y**2) - (-S(1 - y**2))
    right_len = S(1 - y**2) - S(z**2 - y**2)
    return left_len + right_len

def inner_integral2(y):
    part1 = mp.quad(lambda zz: inner_z_le_y(zz, y), [0, y])
    part2 = mp.quad(lambda zz: inner_z_ge_y(zz, y), [y, 1])
    return part1 + part2

V2 = 2 * mp.quad(lambda yy: inner_integral2(yy), [0, 1])
print("Integral 2 (dx dz dy) =", V2)
# %% [4] Integral #3 : Order = dz dx dy
#
# V = 2 ∫_{y=0..1} ∫_{x=-√(1-y²)..√(1-y²)} ∫_{z=0..√(x²+y²)} dz dx dy

def integrand3(x, y):
    return S(x**2 + y**2)

def inner_integral3(y):
    return mp.quad(lambda xx: integrand3(xx, y),
                   [-S(1 - y**2), S(1 - y**2)])

V3 = 2 * mp.quad(lambda yy: inner_integral3(yy), [0, 1])
print("Integral 3 (dz dx dy) =", V3)
