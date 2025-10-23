import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_pi(n_samples):
#"""Estimate pi using Monte Carlo sampling."""
# Generate random points in unit square [0,1] x [0,1]
    x = np.random. uniform (0, 1, n_samples)
    y = np.random.uniform (0, 1, n_samples)

#Check if points fall inside quarter circle
#Circle equation: x^2 + y^2 <= 1
    inside_circle = (x**2 + y**2) <= 1

#Area of quarter circle / Area of square= (pi/4) / 1 = pi/4
#So pi = 4 (points inside) / (total points)
    pi_estimate = 4*np.sum(inside_circle) / n_samples
    return pi_estimate, x, y, inside_circle

#Run experiment
n=10000
pi_est, x, y, inside = monte_carlo_pi(n)
print (f" Estimated pi: {pi_est:.6f}")
print (f"Actual pi:     {np.pi:.6f}")
print (f"Error:         {abs (pi_est-np.pi):.6f}")

#Create visualization
fig, (ax1, ax2) = plt.subplots (1, 2, figsize=(12,5))

#Left: Show the sampling
ax1.scatter (x[inside], y [inside], c='blue', s=1, alpha=0.5, label='Inside')
ax1.scatter (x[~inside], y [~inside], c='red', s=1, alpha=0.5, label='Outside')

#Draw quarter circle
theta= np.linspace (0, np.pi/2, 100)
ax1.plot(np.cos(theta), np.sin(theta), 'black', linewidth=2)
ax1.set_xlim (0,1)
ax1.set_ylim (0,1)
ax1.set_aspect('equal')
ax1.set_title (f'Monte Carlo Sampling (n={n})')
ax1.legend ()

#Right: Show convergence
sample_sizes = np.logspace (1, 5, 50, dtype=int)
estimates = []
errors = []

for size in sample_sizes:
    est, _, _, _= monte_carlo_pi(size)
    estimates.append(est)
    errors.append(abs(est-np.pi))

ax2.loglog (sample_sizes, errors,'b-', linewidth=2)
ax2.set_xlabel ('Number of samples')
ax2.set_ylabel('Absolute error')
ax2.set_title('Convergence Analysis')
ax2.grid (True, alpha=0.3)
plt.show()

def monte_carlo_with_statistics(n_samples, n_experiments=100): 
#"""Run multiple experiments to study statistics."""
    estimates = []

    for _ in range(n_experiments): 
        pi_est, _, _, _= Monte_carlo_pi(n_samples)
    estimates.append(piest)
    estimates = np.array (estimates)    
#Calculate statistics
    mean_estimate = np. mean (estimates)
    std_estimate = np.std (estimates)
#Theoretical standard error
#For Monte Carlo: sigma / sqrt(n)
#Here p = pi/4, variance = 4^2 p(1-p) / n
    p=np.pi/4
    theoretical_std = 4*np.sqrt(p*(1-p) / n_samples)
    print (f" Results from {n_experiments} experiments with {n_samples} samples each:")
    print (f"Mean estimate: {mean_estimate: 6f}")
    print (f"Standard deviation: {std_estimate: 6f}")
    print (f"Theoretical std: {theoretical_std:.6f}")
    print (f"95% confidence interval: [{mean_estimate-1.96*std_estimate:.6f},"f"{meanfor in range(n_experiments): estimate + 1.96 std estimate:.6f}]")
    return estimates

def leibniz_pi(n_terms):
#"""Calculate pi using Leibniz series: pi 4 sum((-1)^n / (2n+1))"""
    result = 0
    for n in range(n_terms):
        result += ((-1)**n) / (2*n + 1) 
    return 4* result

def machin_pi(n_terms):
    """Calculate pi using Machin's formula: pi/4 = 4 arctan(1/5) - arctan(1/239)"""
    def arctan_series(x, n):
        result = 0
        for k in range(n):
            result += ((-1)**k * x**(2*k + 1)) / (2*k + 1)
        return result
    return 4 * (4 * arctan_series(1/5, n_terms) - arctan_series(1/239, n_terms))
#Compare methods
methods=['Monte Carlo', 'Leibniz', 'Machin']
n_values =[10, 100, 1000, 10000] 
for n in n_values:
    mc_pi, _, _, _=monte_carlo_pi(n)
    leib_pi=leibniz_pi(n)
    mach_pi=machin_pi (min (n, 20)) # Machin converges very fast

print (f"\nn = {n}:")
print (f"Monte Carlo: {mc_pi:.8f}, error: {abs(mc_pi-np.pi):.2e}")
print (f" Leibniz:{leib_pi:.8f}, error: {abs(leib_pi-np.pi): 2e}")
print (f" Machin:{mach_pi:.8f}, error: {abs (mach_pi-np.pi): 2e}")