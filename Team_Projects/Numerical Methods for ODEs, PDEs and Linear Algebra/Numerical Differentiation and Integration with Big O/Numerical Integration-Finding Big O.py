"""Composite Simpson convergence study; repaired 2026-10-20.

Use sin(x) on [0, pi], with analytical integral 2. The former script called
an undefined function and supplied odd interval counts to an even-interval
method. Importing the repaired file performs no experiment or plot writes.
"""
import numpy as np


def SimC(F, a, b, N):
    """Composite Simpson rule with N even subintervals (N+1 sample nodes)."""
    if not isinstance(N, (int, np.integer)) or N < 2 or N % 2:
        raise ValueError("N must be an even positive integer >= 2.")
    if not np.isfinite([a,b]).all() or a >= b:
        raise ValueError("Expected finite endpoints a < b.")
    x = np.linspace(a,b,N+1)
    values = np.asarray([F(float(v)) for v in x],dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Integrand values must be finite.")
    return (b-a)/(3*N)*(values[0]+values[-1]+4*values[1:-1:2].sum()+2*values[2:-1:2].sum())


def convergence_study(max_power=9):
    if not isinstance(max_power,int) or not 2 <= max_power <= 16:
        raise ValueError("max_power must be between 2 and 16.")
    intervals = 2**np.arange(1,max_power+1)
    errors = np.array([abs(SimC(np.sin,0,np.pi,int(n))-2.0) for n in intervals])
    orders = np.full(len(intervals),np.nan)
    valid = (errors[:-1]>0)&(errors[1:]>0)
    orders[1:][valid] = np.log2(errors[:-1][valid]/errors[1:][valid])
    return intervals,errors,orders


def main():
    import matplotlib.pyplot as plt
    intervals,errors,orders = convergence_study()
    for n,e,p in zip(intervals,errors,orders):
        print(f"N={n:5d}, absolute error={e:.6e}, observed order={p:.4f}")
    plt.loglog(intervals,errors,"o-",label="Simpson absolute error")
    plt.loglog(intervals,errors[0]*(intervals[0]/intervals)**4,"--",label="N^-4 reference")
    plt.xlabel("Number of subintervals")
    plt.ylabel("Absolute integral error")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
