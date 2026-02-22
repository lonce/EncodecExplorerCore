import torch

@torch.no_grad()
def target2poolindex_cos(
    target: torch.Tensor,        # [T,D] unnormalized ok
    pool_norm: torch.Tensor,      # [P,D] MUST be row-normalized
    window: int = 1,
    eps: float = 1e-8,
):
    """
    Match by cosine similarity (windowed, non-overlapping).
    Returns:
      idx_T:      [T] long
      cos_sim_T:  [T] float, cosine similarity for each target frame vs chosen pool frame
    """
    if target.ndim != 2 or pool_norm.ndim != 2:
        raise ValueError(f"target and pool_norm must be [*,D]. Got {target.shape=} {pool_norm.shape=}")
    if target.shape[1] != pool_norm.shape[1]:
        raise ValueError(f"Dim mismatch: target D={target.shape[1]} vs pool D={pool_norm.shape[1]}")
    if window < 1:
        raise ValueError("window must be >= 1")

    device = target.device
    pool_norm = pool_norm.to(device)

    # normalize target internally for cosine
    target_norm = torch.nn.functional.normalize(target.float(), dim=1, eps=eps).to(device)

    T, D = target_norm.shape
    P = pool_norm.shape[0]
    if window > P:
        raise ValueError(f"window={window} is larger than pool length P={P}")
    if window != 1 and (T % window) != 0:
        print(f"WARNING: window={window} does not evenly divide target length T={T}. "
              f"Handling final remainder of {T % window} frames with a shorter window.")

    pool_T = pool_norm.transpose(0, 1).contiguous()  # [D,P]
    idx_out = torch.empty(T, dtype=torch.long, device=device)

    def best_start_for_window(win_LD: torch.Tensor) -> int:
        L = win_LD.shape[0]
        P_L = P - L + 1
        sims = win_LD @ pool_T  # [L,P]

        base = torch.arange(P_L, device=device).unsqueeze(1)   # [P_L,1]
        offs = torch.arange(L, device=device).unsqueeze(0)     # [1,L]
        idx_mat = base + offs                                  # [P_L,L]

        rows = torch.arange(L, device=device).unsqueeze(0).expand(P_L, L)
        gathered = sims[rows, idx_mat]                         # [P_L,L]
        scores = gathered.sum(dim=1)                           # [P_L]
        return int(scores.argmax().item())

    # fill indices window-by-window
    t = 0
    while t + window <= T:
        win = target_norm[t:t+window]
        p_best = best_start_for_window(win)
        idx_out[t:t+window] = torch.arange(p_best, p_best + window, device=device, dtype=torch.long)
        t += window

    rem = T - t
    if rem > 0:
        win = target_norm[t:T]
        p_best = best_start_for_window(win)
        idx_out[t:T] = torch.arange(p_best, p_best + rem, device=device, dtype=torch.long)

    # per-frame cosine similarities
    cos_sim_T = (target_norm * pool_norm[idx_out]).sum(dim=1)  # [T]
    return idx_out, cos_sim_T

#==============================================
# and using L2 on unnormalized vectors
#==============================================
@torch.no_grad()
def target2poolindex_l2(
    target: torch.Tensor,     # [T,D] unnormalized
    pool: torch.Tensor,       # [P,D] unnormalized
    window: int = 1,
):
    """
    Match by L2 distance (windowed, non-overlapping).
    Returns:
      idx_T:     [T] long
      l2_T:      [T] float, L2 distance per frame vs chosen pool frame (raw space)
    """
    if target.ndim != 2 or pool.ndim != 2:
        raise ValueError(f"target and pool must be [*,D]. Got {target.shape=} {pool.shape=}")
    if target.shape[1] != pool.shape[1]:
        raise ValueError(f"Dim mismatch: target D={target.shape[1]} vs pool D={pool.shape[1]}")
    if window < 1:
        raise ValueError("window must be >= 1")

    device = target.device
    pool = pool.to(device)
    target = target.float().to(device)

    T, D = target.shape
    P = pool.shape[0]
    if window > P:
        raise ValueError(f"window={window} is larger than pool length P={P}")
    if window != 1 and (T % window) != 0:
        print(f"WARNING: window={window} does not evenly divide target length T={T}. "
              f"Handling final remainder of {T % window} frames with a shorter window.")

    idx_out = torch.empty(T, dtype=torch.long, device=device)

    def best_start_for_window_l2(win_LD: torch.Tensor) -> int:
        """
        win_LD: [L,D]
        Find p minimizing sum_i ||win[i] - pool[p+i]||_2
        """
        L = win_LD.shape[0]
        P_L = P - L + 1

        # Build pool windows: pool[p:p+L] for all p as a view via indexing
        # We'll compute distances by gathering aligned frames.
        # For each i, compare win[i] to pool[i : i+P_L]
        total = torch.zeros(P_L, device=device, dtype=win_LD.dtype)

        for i in range(L):
            # pool_slice: [P_L, D] where row p is pool[p+i]
            pool_slice = pool[i:i+P_L]                 # [P_L,D]
            diff = pool_slice - win_LD[i].unsqueeze(0) # [P_L,D]
            total += torch.linalg.norm(diff, dim=1)    # accumulate per-frame L2

        return int(total.argmin().item())

    # fill indices window-by-window
    t = 0
    while t + window <= T:
        win = target[t:t+window]
        p_best = best_start_for_window_l2(win)
        idx_out[t:t+window] = torch.arange(p_best, p_best + window, device=device, dtype=torch.long)
        t += window

    rem = T - t
    if rem > 0:
        win = target[t:T]
        p_best = best_start_for_window_l2(win)
        idx_out[t:T] = torch.arange(p_best, p_best + rem, device=device, dtype=torch.long)

    # per-frame raw L2 distances
    chosen = pool[idx_out]                               # [T,D]
    l2_T = torch.linalg.norm(target - chosen, dim=1)     # [T]
    return idx_out, l2_T



#============================================================
#============================================================
import numpy as np
import matplotlib.pyplot as plt

def plot_match_scores(cos_sim=None, l2_dist=None, fps=None, title="Match scores over time"):
    """
    Plot cosine similarity and/or L2 distance as separate figures.
    - cos_sim: 1D array-like, higher is better
    - l2_dist: 1D array-like, lower is better
    - fps: if provided, x-axis is seconds; otherwise frames
    """
    if cos_sim is None and l2_dist is None:
        raise ValueError("Provide at least one of cos_sim or l2_dist.")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive.")

    if cos_sim is not None:
        cos_sim = np.asarray(cos_sim, dtype=float).reshape(-1)
        x = np.arange(len(cos_sim), dtype=float)
        xlabel = "Frame"
        if fps is not None:
            x = x / float(fps)
            xlabel = "Time (s)"
        plt.figure()
        plt.plot(x, cos_sim)
        plt.xlabel(xlabel)
        plt.ylabel("Cosine similarity (higher is better)")
        plt.title(title + " — Cosine similarity")
        plt.show()

    if l2_dist is not None:
        l2_dist = np.asarray(l2_dist, dtype=float).reshape(-1)
        x = np.arange(len(l2_dist), dtype=float)
        xlabel = "Frame"
        if fps is not None:
            x = x / float(fps)
            xlabel = "Time (s)"
        plt.figure()
        plt.plot(x, l2_dist)
        plt.xlabel(xlabel)
        plt.ylabel("L2 distance (lower is better)")
        plt.title(title + " — L2 distance")
        plt.show()


def plot_match_scores_normalized(cos_sim=None, l2_dist=None, fps=None, title="Normalized match scores over time"):
    """
    Plot both metrics on a comparable [0,1] scale (higher is better), as separate figures.
    - cos_sim is min-max normalized to [0,1]
    - l2_dist is inverted (lower better) then min-max normalized: score = -l2_dist
    """
    if cos_sim is None and l2_dist is None:
        raise ValueError("Provide at least one of cos_sim or l2_dist.")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive.")

    def _minmax(a):
        a = np.asarray(a, dtype=float).reshape(-1)
        mn = float(np.min(a))
        mx = float(np.max(a))
        if mx - mn < 1e-12:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)

    if cos_sim is not None:
        cos_sim = np.asarray(cos_sim, dtype=float).reshape(-1)
        cos01 = _minmax(cos_sim)
        x = np.arange(len(cos01), dtype=float)
        xlabel = "Frame"
        if fps is not None:
            x = x / float(fps)
            xlabel = "Time (s)"
        plt.figure()
        plt.plot(x, cos01)
        plt.xlabel(xlabel)
        plt.ylabel("Normalized score (higher is better)")
        plt.title(title + " — Cosine similarity (min-max)")
        plt.show()

    if l2_dist is not None:
        l2_dist = np.asarray(l2_dist, dtype=float).reshape(-1)
        inv01 = _minmax(-l2_dist)  # invert so higher is better
        x = np.arange(len(inv01), dtype=float)
        xlabel = "Frame"
        if fps is not None:
            x = x / float(fps)
            xlabel = "Time (s)"
        plt.figure()
        plt.plot(x, inv01)
        plt.xlabel(xlabel)
        plt.ylabel("Normalized score (higher is better)")
        plt.title(title + " — L2 distance (inverted, min-max)")
        plt.show()
