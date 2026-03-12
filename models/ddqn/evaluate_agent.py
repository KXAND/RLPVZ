import numpy as np


def evaluate(env, agent, n_iter=1000, verbose=True):
    sum_score = 0
    sum_iter = 0
    score_hist = []
    iter_hist = []

    for episode_idx in range(n_iter):
        if verbose:
            print("\r{}/{}".format(episode_idx, n_iter), end="")

        # play episodes
        summary = env.play(agent)
        summary["score"] = np.sum(summary["rewards"])

        score_hist.append(summary["score"])
        max_frames = getattr(env.env, "max_steps", None)
        steps = getattr(env.env, "steps", 0)
        iter_hist.append(min(steps, max_frames) if max_frames else steps)

        sum_score += summary["score"]
        sum_iter += min(steps, max_frames) if max_frames else steps

    if verbose:
        avg_score = sum_score / n_iter
        avg_iter = sum_iter / n_iter
        print(f"\nEval avg_score={avg_score:.2f}, avg_iter={avg_iter:.2f}")

    return sum_score / n_iter, sum_iter / n_iter
