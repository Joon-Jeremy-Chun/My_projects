# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:11:08 2024

@author: joonc
"""

from fractions import Fraction

# Convert to fractions for precise results
P_W_given_H_frac = Fraction(5, 12)
P_W_given_T_frac = Fraction(1, 5)
P_H_frac = Fraction(1, 2)
P_T_frac = Fraction(1, 2)

# Law of total probability for P(W) in fractions
P_W_frac = (P_W_given_H_frac * P_H_frac) + (P_W_given_T_frac * P_T_frac)

# Bayes' Theorem to find P(T | W) in fractions
P_T_given_W_frac = (P_W_given_T_frac * P_T_frac) / P_W_frac

P_T_given_W_frac
print(P_T_given_W_frac)