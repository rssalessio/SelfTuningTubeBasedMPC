
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import plot_constants
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


volume = np.load('delta_data.npy')
DELTAS = [1e-4, 1e-3, 1e-2, 1e-1]


linestyle_str = [
     'solid',      # Same as (0, ()) or '-'
     'dotted',    # Same as (0, (1, 1)) or ':'
     'dashed',    # Same as '--'
     'dashdot',]  # Same as '-.'

mu = volume.mean(axis=0)

mu = mu / mu[:,0][:, None]
t = [2, 6, 16, 51, 101]

for idx_delta, delta in enumerate(DELTAS):
    for step in t:
        print(f'[Step {step-1} - Delta: {delta}] Volume w.r.t. Theta0 {100*mu[idx_delta][step]}')


plt.figure(figsize=(8,5))
for idx_delta, delta in enumerate(DELTAS):
    mu = volume[:, idx_delta, :].mean(axis=0)
    std = volume[:, idx_delta, :].std(axis=0)
    x = np.arange(volume.shape[-1])
    plt.plot(x,mu, label=f'E[Vol($\Theta_t$)] - $\delta={delta}$', linestyle=linestyle_str[idx_delta])
    plt.fill_between(x, mu - 4.5 * std / np.sqrt(volume.shape[0]),  mu + 4.5* std / np.sqrt(volume.shape[0]), alpha=0.3)

plt.xlabel('Time $t$')
plt.ylabel('E[Vol($\Theta_t$)]')
plt.legend()
plt.grid()


plt.yscale('log')
plt.tight_layout()
plt.show()

