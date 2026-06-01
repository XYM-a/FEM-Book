import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wynn_epsilon(sequence):
    """
    Wynn-epsilon extrapolation for a scalar sequence.

    eps[-1, n] = 0
    eps[ 0, n] = s_n
    eps[k+1, n] = eps[k-1, n+1] + 1 / (eps[k, n+1] - eps[k, n])
    """
    sequence = np.asarray(sequence, dtype=float)
    N = len(sequence)

    eps = {(-1, i): 0.0 for i in range(N)}
    eps.update({(0, i): float(sequence[i]) for i in range(N)})

    for k in range(N - 1):
        for i in range(N - k - 1):
            diff = eps[(k, i + 1)] - eps[(k, i)]

            if abs(diff) < 1e-300:
                eps[(k + 1, i)] = np.nan
            else:
                eps[(k + 1, i)] = eps[(k - 1, i + 1)] + 1.0 / diff

    return eps


# ============================================================
# 1. Direct polygon approximation
#    pi_n = n sin(pi/n)
#    h = 1/n
# ============================================================

n = 2 ** np.arange(0, 9)      # n = 1, 2, 4, ..., 256
h = 1.0 / n

pi_direct = n * np.sin(np.pi / n)
error_direct = np.abs(np.pi - pi_direct)


# ============================================================
# 2. Wynn-epsilon extrapolation
# ============================================================

eps = wynn_epsilon(pi_direct)

# Same extrapolated sequence levels as the lecture-table layout:
# epsilon_2^(0), epsilon_4^(0), epsilon_6^(0), epsilon_8^(0)
n_extra = np.array([4, 16, 64, 256])
h_extra = 1.0 / n_extra

pi_extra = np.array([
    eps[(2, 0)],
    eps[(4, 0)],
    eps[(6, 0)],
    eps[(8, 0)]
])

error_extra = np.abs(np.pi - pi_extra)


# ============================================================
# 3. Observed convergence rates
#    rate = log(e_i/e_{i+1}) / log(h_i/h_{i+1})
# ============================================================

rate_direct = np.log(error_direct[:-1] / error_direct[1:]) / np.log(h[:-1] / h[1:])
rate_extra = np.log(error_extra[:-1] / error_extra[1:]) / np.log(h_extra[:-1] / h_extra[1:])


print("Direct approximation:")
for i in range(len(n)):
    print(f"n={n[i]:3d}, h={h[i]:.8f}, pi_n={pi_direct[i]:.16f}, error={error_direct[i]:.4e}")

print("\nObserved convergence rates for direct approximation:")
print(np.round(rate_direct, 4))

print("\nWynn-epsilon extrapolation:")
for i in range(len(n_extra)):
    print(f"n={n_extra[i]:3d}, h={h_extra[i]:.8f}, pi_ext={pi_extra[i]:.16f}, error={error_extra[i]:.4e}")

print("\nObserved convergence rates for Wynn-epsilon extrapolation:")
print(np.round(rate_extra, 4))


# ============================================================
# 4. Save numerical data
# ============================================================

rows = []

for i, ni in enumerate(n):
    rows.append({
        "method": "direct",
        "n": int(ni),
        "h": h[i],
        "pi_approx": pi_direct[i],
        "abs_error": error_direct[i],
        "observed_rate_to_next": rate_direct[i] if i < len(rate_direct) else np.nan
    })

for i, ni in enumerate(n_extra):
    rows.append({
        "method": "Wynn-epsilon",
        "n": int(ni),
        "h": h_extra[i],
        "pi_approx": pi_extra[i],
        "abs_error": error_extra[i],
        "observed_rate_to_next": rate_extra[i] if i < len(rate_extra) else np.nan
    })

df = pd.DataFrame(rows)
df.to_csv("pi_convergence_true_reproduction_data.csv", index=False)


# ============================================================
# 5. Plot
# ============================================================

fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=180)

ax.loglog(
    h,
    error_direct,
    marker="v",
    linewidth=1.5,
    markersize=5,
    label=r"Direct: $\pi_n=n\sin(\pi/n)$"
)

ax.loglog(
    h_extra,
    error_extra,
    marker="^",
    linewidth=1.5,
    markersize=5,
    label=r"Wynn-$\epsilon$ extrapolation"
)

ax.set_xlabel(r"$h=1/n$")
ax.set_ylabel(r"$e_n=|\pi-\pi_n|$")

ax.set_xlim(1e-3, 1)
ax.set_ylim(1e-15, 1e1)

ax.grid(True, which="both", linestyle=":", linewidth=0.6)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("pi_convergence_true_reproduction.png", bbox_inches="tight")
plt.show()