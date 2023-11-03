# -*- coding: utf-8 -*-

import math

#hybrid method = bisection + secant
#hybrid function input define
# 1.f; function, for radious 2.g; function, for hight 3. a; left point 4. b; right point
# 5. diff_a_b; in this cods difference between x_n_+1 and x_n 6. max_iter; bounced max iteration
# note - we are define the error (accuracy) bounced by diff_a_b. because it is cauchy sequence.

#hybrid function output define 
# 1. r; radious 2. iteration 3. h; hight

def hybrid(f, g, a, b, diff_a_b, max_iter):
# Condition to proceed with iterations
    if f(a) * f(b) >= 0:
        print("hybrid method may not converge because f(a) * f(b) >= 0")
        return None
    
    if f(b) - f(a) == 0 :
        print("hybrid method may not converge becasue f(b) - f(a) = 0")
        return None

    iteration = 0
    x_prime = (a+b)/2 #initiated the first sudo x_prime (not record, not effect) for geting start the first roof
    
    while (x_prime - (b - ((f(b)*(b-a))/(f(b)-f(a)))) ) > diff_a_b and iteration < max_iter and f(b) - f(a) != 0:
        # Perform secant method first
        x_prime = b - ((f(b)*(b-a))/(f(b)-f(a)))

        #Denote iteration
        iteration += 1
        #print("~~iteration :", iteration)

        #Check for location of new x value
        #First condition checks if new value lies outside of the interval
        if x_prime < a or x_prime>b:
            #Performs bisection to obtain new x value
            #print("bicetion mothod")
            x_prime = (b-a)/2.0

            #Checks image of new x value
            if f(a) * f(x_prime) < 0:
                b = x_prime
            #    print("a:", a, "b:", x_prime)
            else:
                a = x_prime
            #    print("a:", x_prime, "b:", b)

        else:
            #Checks image of new x value if secant lies within the interval
            #print("secant mothod")
            if f(a)*f(x_prime)<0:
                b = x_prime
            #    print("a:", a, "b:", x_prime)
            else:
                a=x_prime
            #    print("a:", x_prime, "b:", b)

        #print("\n")
    height = g(x_prime)
    return x_prime, iteration, height

# Define the function that want to find the root of
def f(x):
    return (-2000/(x**2))-(500/((math.pi)*(x**3)))+(4*math.pi*x)+math.pi
def g(c):
    return 1000/(math.pi*c**2)

a = 5 
b = 6
diff_a_b = 1e-4
# max_iterations = 100
#Print final number of iterations required to land within the error
result = hybrid(f, g, a, b, diff_a_b, 100)
print(result)

if result[1]<100:
    print(f"We require {result[1]} iterations for the error to lie within 10e-4.")
else:
    print("We require over 100 iterations for the error to lie within 10e-4.")
print(f"The most optimal radius of the can will be {round(result[0],2)} cm and the most optimal height will be  {round(result[2],2)} cm.")