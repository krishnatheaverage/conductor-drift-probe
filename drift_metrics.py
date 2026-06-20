"""
drift_metrics.py  --  scientific core for the CONDUCTOR drift-probe experiment.

The load-bearing premise of the CONDUCTOR framework is that, in multi-turn
dialogue, the activation-space subspace that encodes safety / refusal (S_t) and
the subspace that encodes hallucination / epistemic-uncertainty (H_t) ROTATE as
turns accumulate, so a single static disentangling projection re-entangles or
over-projects later in a conversation.

This module contains only the geometry and the null-model statistics. It has NO
model dependency (pure numpy), so it is exercised verbatim by both the synthetic
self-test (test_synthetic.py, runs anywhere) and the real-model run
(run_drift_probe.py). If the math is right in the synthetic test, it is right in
the real run, because it is the same code.

Key design choices, and why:

  * Drift is measured as the CHORDAL distance between consecutive-turn subspaces,
    derived from principal angles. This is GAUGE-INVARIANT (independent of the
    arbitrary basis returned by the estimator), so we never need a Procrustes
    alignment and never confound "the basis spun" with "the subspace moved".

  * Everything is normalized to [0, 1]. d_norm = 0 means identical subspaces,
    d_norm = 1 means fully orthogonal. Same for the cross-subspace overlap.
    This makes "non-trivial" interpretable rather than scale-dependent.

  * "Non-trivial" is defined ONLY relative to a null. Estimation noise makes raw
    drift strictly positive even for a perfectly static subspace, so the headline
    statistic is the ratio (observed consecutive drift) / (within-turn bootstrap
    noise floor), with bootstrap confidence intervals on both.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Subspace estimation from contrastive activations                            #
# --------------------------------------------------------------------------- #

def orthonormal_basis(M: np.ndarray, k: int) -> np.ndarray:
    """Top-k left singular vectors of M (d x n), returned as a d x k orthonormal basis."""
    # Economy SVD; columns of U are orthonormal directions ordered by singular value.
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    if U.shape[1] < k:
        raise ValueError(f"cannot extract {k} directions from rank-{U.shape[1]} signal")
    return U[:, :k]


def subspace_from_contrast(acts_pos: np.ndarray, acts_neg: np.ndarray, k: int) -> np.ndarray:
    """
    Estimate the k-dimensional subspace that distinguishes the positive class from
    the negative class, from PAIRED contrastive activations.

    acts_pos, acts_neg : (n, d) arrays of last-token activations for n matched
                         probe pairs (e.g. harmful vs harmless request; or
                         unknown-entity vs known-entity question).

    Method: difference-of-class signal. We form the per-pair difference vectors
    and the class-mean difference, stack them, mean-center, and take the top-k
    principal directions (left singular vectors). For k=1 this reduces to the
    standard difference-of-means direction used for refusal/truth probing
    (Arditi et al. 2024; Marks & Tegmark 2023).
    """
    acts_pos = np.asarray(acts_pos, dtype=np.float64)
    acts_neg = np.asarray(acts_neg, dtype=np.float64)
    if acts_pos.shape != acts_neg.shape:
        raise ValueError("paired contrast requires equal-shape pos/neg activation sets")

    diffs = acts_pos - acts_neg                      # (n, d) per-pair contrast
    # Top-k directions of contrast ENERGY (uncentered PCA of the paired differences).
    # The leading direction is the difference-of-means (the standard refusal/truth
    # probe direction; Arditi et al. 2024, Marks & Tegmark 2023) whenever a consistent
    # class shift exists, and subsequent directions capture consistent secondary axes
    # of separation. Crucially this does NOT mix in centered per-pair noise, which
    # would dominate the top-k at low SNR (a bug the synthetic self-test caught).
    return orthonormal_basis(diffs.T, k)             # d x k


# --------------------------------------------------------------------------- #
# Subspace geometry (gauge-invariant)                                         #
# --------------------------------------------------------------------------- #

def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between subspaces col(U) and col(V)."""
    # Singular values of U^T V are the cosines of the principal angles.
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def subspace_drift(U: np.ndarray, V: np.ndarray) -> float:
    """
    Normalized chordal distance in [0, 1] between two k-planes.

    d_norm = sqrt( mean_i sin^2(theta_i) ) = ||P_U - P_V||_F / sqrt(2k).
    0  -> identical subspaces;  1 -> fully orthogonal.
    Gauge-invariant: depends only on the subspaces, not their chosen bases.
    """
    ang = principal_angles(U, V)
    return float(np.sqrt(np.mean(np.sin(ang) ** 2)))


def cumulative_displacement(U_list: list[np.ndarray], base: int = 0) -> list[float]:
    """
    Chordal distance from a baseline turn to each later turn's subspace:
    [ d(U_base, U_t) for t = base+1 .. T-1 ].

    More sensitive than consecutive-turn drift to SLOW accumulation: many per-turn
    steps can each sit near the noise floor while the subspace still walks
    progressively away from its baseline.

    IMPORTANT (confound control): use base=1, not base=0, for the premise test.
    Turn 0 has NO context, so d(U_0, U_t) is dominated by the generic "no-context ->
    has-context" representational jump, which is shared by every dialogue and is not
    the multi-turn ACCUMULATION the framework is about. Baselining at turn 1 (the
    first contextful turn) measures context-to-more-context drift. The turn0->turn1
    onset should be reported separately as an uninteresting effect.
    """
    return [subspace_drift(U_list[base], U_list[t]) for t in range(base + 1, len(U_list))]


def overlap_omega(U_s: np.ndarray, U_h: np.ndarray) -> float:
    """
    Normalized cross-subspace overlap (entanglement) in [0, 1] between the safety
    subspace and the hallucination subspace.

    omega = ||U_s^T U_h||_F^2 / k = mean_i cos^2(theta_i^{SH}).
    1 -> subspaces coincide (maximal entanglement);  0 -> orthogonal (disentangled).
    This is the quantity a static orthogonalizer drives toward 0 once, globally;
    the CONDUCTOR premise is that it MOVES across turns.
    """
    k = U_s.shape[1]
    return float(np.linalg.norm(U_s.T @ U_h, "fro") ** 2 / k)


# --------------------------------------------------------------------------- #
# Null models                                                                 #
# --------------------------------------------------------------------------- #

def bootstrap_noise_floor(
    acts_pos: np.ndarray,
    acts_neg: np.ndarray,
    k: int,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Within-turn estimation-noise floor for CONSECUTIVE-turn drift.

    The right null for "drift between turn t-1 and turn t under NO real change" is
    the drift between two INDEPENDENT noisy estimates of the same (unchanged)
    subspace. So each iteration draws two independent bootstrap resamples of the
    probe pairs, estimates a subspace from each, and measures the drift between
    them. This is directly comparable to an observed consecutive-turn drift (both
    are "distance between two independent estimates"); a replicate-vs-full design
    would understate it by ~sqrt(2) and make a static subspace look like motion.
    """
    rng = rng or np.random.default_rng(0)
    n = acts_pos.shape[0]
    dists = np.empty(n_boot)
    for b in range(n_boot):
        i1 = rng.integers(0, n, size=n)
        i2 = rng.integers(0, n, size=n)
        U1 = subspace_from_contrast(acts_pos[i1], acts_neg[i1], k)
        U2 = subspace_from_contrast(acts_pos[i2], acts_neg[i2], k)
        dists[b] = subspace_drift(U1, U2)
    return _summ(dists)


def noise_corrected_drift(d_meas: float, floor_median: float) -> float:
    """
    Remove the upward bias from estimation noise. If two independent estimates of
    the SAME subspace drift by ~floor, then an observed drift d_meas between two
    turns decomposes (in the small-angle regime) as d_meas^2 ~ delta^2 + floor^2,
    where delta is the true rotation. Return the de-biased estimate of delta,
    floored at 0. This is approximate (small-angle, isotropic-error) and is
    reported alongside, never instead of, the raw drift and the significance test.
    """
    v = d_meas ** 2 - floor_median ** 2
    return float(np.sqrt(v)) if v > 0 else 0.0


def overlap_noise_floor(
    pos_s: np.ndarray, neg_s: np.ndarray,
    pos_h: np.ndarray, neg_h: np.ndarray,
    k: int, n_boot: int = 200, rng: np.random.Generator | None = None,
) -> dict:
    """Bootstrap noise band for the cross-subspace overlap omega at a fixed turn."""
    rng = rng or np.random.default_rng(1)
    n = pos_s.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        U_s = subspace_from_contrast(pos_s[idx], neg_s[idx], k)
        U_h = subspace_from_contrast(pos_h[idx], neg_h[idx], k)
        vals[b] = overlap_omega(U_s, U_h)
    return _summ(vals)


def _summ(x: np.ndarray) -> dict:
    return {
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "ci_lo": float(np.percentile(x, 2.5)),
        "ci_hi": float(np.percentile(x, 97.5)),
        "n": int(x.size),
    }


# --------------------------------------------------------------------------- #
# Pre-registered decision rule                                                #
# --------------------------------------------------------------------------- #

def decide(
    consec_drift: list[float],
    noise_floor: dict,
    omega_series: list[float],
    omega_noise: dict,
    min_effect: float = 0.05,
) -> dict:
    """
    Pre-registered verdict on whether the CONDUCTOR premise survives. The verdict
    is mechanical (no judgement), and rests on a significance test plus an effect
    size, not an arbitrary ratio.

    H1 (premise SUPPORTED): the subspaces rotate across turns, i.e.
        (a) SIGNIFICANCE: median consecutive-turn drift exceeds the upper 95%
            bound of the no-change bootstrap null (bands disjoint), AND
        (b) EFFECT SIZE: the noise-corrected drift exceeds min_effect (a rotation
            large enough to matter, default chordal 0.05 ~ 5 deg of a single
            principal angle on a rank-3 subspace), AND
        (c) the cross-subspace overlap omega ranges across turns by more than its
            own bootstrap noise band (the entanglement geometry actually moves).

    H0 (premise FALSIFIED -> a STATIC projection suffices, do not build the
        framework): drift is within the no-change null, or the corrected effect is
        below min_effect, or omega is flat within its noise band.
    """
    consec = np.asarray(consec_drift, dtype=np.float64)
    med = float(np.median(consec))
    floor_med = float(noise_floor["median"])
    drift_ratio = med / max(floor_med, 1e-12)
    bands_disjoint = med > noise_floor["ci_hi"]
    corrected = noise_corrected_drift(med, floor_med)
    effect_ok = corrected >= min_effect

    omega = np.asarray(omega_series, dtype=np.float64)
    omega_range = float(omega.max() - omega.min())
    omega_band = float(omega_noise["ci_hi"] - omega_noise["ci_lo"])
    omega_moves = omega_range > omega_band

    supported = bool(bands_disjoint and effect_ok and omega_moves)

    return {
        "verdict": "PREMISE_SUPPORTED" if supported else "PREMISE_FALSIFIED",
        "drift_median": med,
        "drift_noise_floor_median": floor_med,
        "drift_noise_floor_ci_hi": float(noise_floor["ci_hi"]),
        "drift_ratio": drift_ratio,
        "drift_median_noise_corrected": corrected,
        "drift_bands_disjoint": bands_disjoint,
        "effect_exceeds_min": effect_ok,
        "min_effect_required": min_effect,
        "omega_range_across_turns": omega_range,
        "omega_noise_band_width": omega_band,
        "omega_moves": omega_moves,
        "note": (
            "PREMISE_SUPPORTED means the subspaces measurably rotate across turns, "
            "so a static projection is mis-specified and the conditional CONDUCTOR "
            "machinery is motivated. PREMISE_FALSIFIED means a static projection "
            "suffices and the framework should not be built. If FALSIFIED with a "
            "high noise floor, the run is underpowered: add probe pairs and retry "
            "before concluding."
        ),
    }
