"""
test_synthetic.py  --  validate the drift estimator and decision rule against
ground truth, with NO model and NO network. Runs anywhere.

We construct paired activations whose contrast subspace is a KNOWN, analytically
controlled object, then confirm:

  (1) the estimator recovers the injected subspace at high SNR;
  (2) a STATIC true subspace -> measured drift collapses to the bootstrap noise
      floor -> decision rule returns PREMISE_FALSIFIED;
  (3) a subspace ROTATING by a known per-turn angle -> measured drift recovers the
      injected angle, clears the noise floor -> rule returns PREMISE_SUPPORTED, and
      a cross-subspace overlap with a known time-varying relative angle is tracked;
  (4) principal-angle sanity (identical -> 0, orthogonal -> pi/2).

If these pass, the geometry, null model, and decision rule are correct, so any
verdict from the real run reflects the data, not a code artifact.
"""

import numpy as np

from drift_metrics import (
    subspace_from_contrast, subspace_drift, principal_angles, overlap_omega,
    bootstrap_noise_floor, overlap_noise_floor, decide,
)

D = 256          # ambient activation dimension
K = 3            # subspace rank
N = 96           # probe pairs per turn (enough to push the noise floor below the signal)
T = 8            # dialogue turns
rng = np.random.default_rng(7)


def basis_from_dirs(dirs):
    """Orthonormalize a list of d-vectors into a d x k basis."""
    M = np.stack(dirs, axis=1)
    Q, _ = np.linalg.qr(M)
    return Q[:, :len(dirs)]


def rotated_subspace(t, phi, plane=(0, K), fixed=(1, 2)):
    """
    A K-plane whose FIRST direction rotates by angle phi*t in the (e_a, e_b) plane,
    with the remaining directions held fixed. Consecutive turns then differ by a
    single principal angle = phi, the rest 0  =>  analytic drift d_norm = sin(phi)/sqrt(K).
    """
    a, b = plane
    d0 = np.zeros(D); d0[a] = np.cos(phi * t); d0[b] = np.sin(phi * t)
    dirs = [d0] + [np.eye(D)[i] for i in fixed]
    return basis_from_dirs(dirs)


def overlap_pair(t, phi_s, phi_h):
    """
    S_t and H_t whose leading directions sit in the shared (e0,e1) plane at angles
    phi_s*t and phi_h*t, with disjoint fixed tails. Then normalized overlap is
    cos^2((phi_s-phi_h) t) / K, which moves across turns by construction.
    """
    s0 = np.zeros(D); s0[0] = np.cos(phi_s * t); s0[1] = np.sin(phi_s * t)
    h0 = np.zeros(D); h0[0] = np.cos(phi_h * t); h0[1] = np.sin(phi_h * t)
    S = basis_from_dirs([s0, np.eye(D)[2], np.eye(D)[3]])
    H = basis_from_dirs([h0, np.eye(D)[4], np.eye(D)[5]])
    return S, H


def make_turn_acts(U_true, signal_strength=8.0, noise=0.3):
    """Paired pos/neg activations whose contrast energy lies in col(U_true)."""
    coeffs = rng.normal(size=(N, U_true.shape[1])) * signal_strength
    base = rng.normal(size=(N, D)) * noise            # shared context, cancels in the difference
    pos = base + coeffs @ U_true.T + rng.normal(size=(N, D)) * noise
    neg = base + rng.normal(size=(N, D)) * noise
    return pos, neg


def report(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    print("Synthetic validation of drift_metrics.py")
    all_ok = True

    # --- (1) estimator fidelity --------------------------------------------
    print("\n(1) estimator recovers an injected subspace at high SNR")
    U_true, _ = np.linalg.qr(rng.normal(size=(D, K)))
    pos, neg = make_turn_acts(U_true)
    U_hat = subspace_from_contrast(pos, neg, K)
    recon_err = subspace_drift(U_true, U_hat)
    all_ok &= report(f"recovery chordal error {recon_err:.3f} < 0.10", recon_err < 0.10)

    # --- (2) STATIC truth -> falsified -------------------------------------
    print("\n(2) zero true rotation -> drift ~ noise floor -> PREMISE_FALSIFIED")
    Us = [rotated_subspace(t, phi=0.0) for t in range(T)]
    acts = [make_turn_acts(U) for U in Us]
    Uhat = [subspace_from_contrast(p, n, K) for p, n in acts]
    drift = [subspace_drift(Uhat[t - 1], Uhat[t]) for t in range(1, T)]
    nf = bootstrap_noise_floor(*acts[0], K, n_boot=150, rng=np.random.default_rng(11))
    Hs = [rotated_subspace(t, phi=0.0, plane=(4, 5), fixed=(6, 7)) for t in range(T)]
    acts_h = [make_turn_acts(U) for U in Hs]
    Uhat_h = [subspace_from_contrast(p, n, K) for p, n in acts_h]
    omega = [overlap_omega(Uhat[t], Uhat_h[t]) for t in range(T)]
    onf = overlap_noise_floor(*acts[0], *acts_h[0], K, n_boot=150, rng=np.random.default_rng(12))
    d_static = decide(drift, nf, omega, onf)
    print(f"      drift median={d_static['drift_median']:.3f}  noise={nf['median']:.3f}  "
          f"ratio={d_static['drift_ratio']:.2f}")
    all_ok &= report("verdict == PREMISE_FALSIFIED for static subspaces",
                     d_static["verdict"] == "PREMISE_FALSIFIED")

    # --- (3) ROTATING truth -> supported -----------------------------------
    print("\n(3) non-zero true rotation -> drift recovers angle -> PREMISE_SUPPORTED")
    phi = np.deg2rad(15.0)
    Us = [rotated_subspace(t, phi=phi) for t in range(T)]
    acts = [make_turn_acts(U) for U in Us]
    Uhat = [subspace_from_contrast(p, n, K) for p, n in acts]
    drift = [subspace_drift(Uhat[t - 1], Uhat[t]) for t in range(1, T)]
    true_drift_med = np.sin(phi) / np.sqrt(K)         # analytic per-turn drift
    nf = bootstrap_noise_floor(*acts[0], K, n_boot=150, rng=np.random.default_rng(13))
    # cross-overlap with a known time-varying relative angle
    SH = [overlap_pair(t, phi_s=np.deg2rad(15.0), phi_h=np.deg2rad(3.0)) for t in range(T)]
    acts_s = [make_turn_acts(S) for S, _ in SH]
    acts_h = [make_turn_acts(H) for _, H in SH]
    Uhat_s = [subspace_from_contrast(p, n, K) for p, n in acts_s]
    Uhat_h = [subspace_from_contrast(p, n, K) for p, n in acts_h]
    omega = [overlap_omega(Uhat_s[t], Uhat_h[t]) for t in range(T)]
    onf = overlap_noise_floor(*acts_s[0], *acts_h[0], K, n_boot=150, rng=np.random.default_rng(14))
    d_rot = decide(drift, nf, omega, onf)
    corrected = d_rot["drift_median_noise_corrected"]
    print(f"      measured drift median={np.median(drift):.3f}  noise-corrected={corrected:.3f}  "
          f"analytic={true_drift_med:.3f}  noise floor={nf['median']:.3f}")
    print(f"      omega range across turns={d_rot['omega_range_across_turns']:.3f}  "
          f"noise band={d_rot['omega_noise_band_width']:.3f}")
    all_ok &= report("noise-corrected drift tracks analytic rotation (within 25%)",
                     abs(corrected - true_drift_med) < 0.25 * true_drift_med)
    all_ok &= report("verdict == PREMISE_SUPPORTED for rotating subspaces",
                     d_rot["verdict"] == "PREMISE_SUPPORTED")

    # --- (4) principal-angle sanity ----------------------------------------
    print("\n(4) principal angles: identical->0, orthogonal->pi/2")
    A = np.eye(D)[:, :K]
    B = np.eye(D)[:, K:2 * K]
    all_ok &= report("identical subspaces -> angle 0",
                     np.allclose(principal_angles(A, A), 0, atol=1e-6))
    all_ok &= report("orthogonal subspaces -> angle pi/2",
                     np.allclose(principal_angles(A, B), np.pi / 2, atol=1e-6))

    print("\n" + ("ALL SYNTHETIC TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
