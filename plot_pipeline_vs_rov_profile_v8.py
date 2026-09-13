"""
plot_pipeline_vs_rov_profile_v8.py
------------------------------------
Compares the pipeline's reference profile against the ROV's actually-flown
profile from the canonical evaluation trajectory (evaluation/eval_trajectory.csv,
produced by `rov_pipeline_rl_v8.py --evaluate`), and reports the tracking
error (RMSE, mean, max).

"Error" here is the same quantity the reward function and the 1m-gap safety
check both use: the 3D Euclidean distance from the ROV's position to the
nearest point on the pipeline centerline (dist_to_pipe_m in the trajectory
CSV, computed via the exact same piecewise-linear projection as training/
evaluation -- not re-derived approximately here).

Output: evaluation/pipeline_vs_rov_profile.png, plus RMSE/mean/max printed
to the console and saved in evaluation/tracking_error_stats.json.

Usage:
    python plot_pipeline_vs_rov_profile_v8.py
    python plot_pipeline_vs_rov_profile_v8.py --csv evaluation/eval_trajectory.csv
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rov_pipeline_rl_v8 import SCENE

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "evaluation")


def load_trajectory(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
    return cols


def pipeline_arclength_profile(pts):
    """Pipeline centerline points -> cumulative arc length s and matching xyz,
    for plotting the reference profile against along-track distance."""
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    return s, pts


def pipeline_pose(pts):
    """Per-point (x,y,z,roll,pitch,yaw) for the static pipeline centerline.
    A cylindrical pipe has no meaningful roll about its own axis, so roll is
    always 0; yaw/pitch come from the direction of the segment leaving each
    point (the last point reuses the final segment's direction)."""
    seg_dir = np.diff(pts, axis=0)
    seg_dir = seg_dir / np.linalg.norm(seg_dir, axis=1, keepdims=True)
    dirs = np.vstack([seg_dir, seg_dir[-1]])  # one direction per point (N,3)
    yaw = np.arctan2(dirs[:, 1], dirs[:, 0])
    horiz = np.sqrt(dirs[:, 0] ** 2 + dirs[:, 1] ** 2)
    pitch = np.arctan2(dirs[:, 2], horiz)
    roll = np.zeros(len(pts))
    return roll, pitch, yaw


def save_pose_csvs(pipe_pts, traj, out_dir):
    """Saves x,y,z,roll,pitch,yaw for the pipeline centerline points and for
    the ROV's full flown trajectory, as two separate CSVs (different native
    sampling: 7 pipeline waypoints vs. one row per simulation step)."""
    roll, pitch, yaw = pipeline_pose(pipe_pts)
    s, _ = pipeline_arclength_profile(pipe_pts)

    pipe_path = os.path.join(out_dir, "pipeline_pose.csv")
    with open(pipe_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point_index", "along_track_m", "x", "y", "z",
                    "roll_deg", "pitch_deg", "yaw_deg"])
        for i in range(len(pipe_pts)):
            w.writerow([i, s[i], *pipe_pts[i],
                        np.degrees(roll[i]), np.degrees(pitch[i]), np.degrees(yaw[i])])
    print(f"Saved pipeline pose to {pipe_path}")

    rov_path = os.path.join(out_dir, "rov_pose.csv")
    with open(rov_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"])
        for i in range(len(traj["t_s"])):
            w.writerow([traj["t_s"][i], traj["x"][i], traj["y"][i], traj["z"][i],
                        np.degrees(traj["roll"][i]), np.degrees(traj["pitch"][i]),
                        np.degrees(traj["yaw"][i])])
    print(f"Saved ROV pose to {rov_path}")


def main(csv_path, out_prefix):
    traj = load_trajectory(csv_path)
    pipe_pts = SCENE["pipeline_points"].astype(np.float64)
    pipe_s, pipe_xyz = pipeline_arclength_profile(pipe_pts)

    error = traj["dist_to_pipe_m"]  # 3D distance from ROV to pipeline centerline, per step
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    max_err = float(np.max(error))
    mean_err = float(np.mean(error))
    std_err = float(np.std(error))

    print("=== Pipeline-tracking error (ROV position vs pipeline centerline) ===")
    print(f"  n samples     : {len(error)}")
    print(f"  mean error    : {mean_err:.4f} m")
    print(f"  std error     : {std_err:.4f} m")
    print(f"  MAE           : {mae:.4f} m")
    print(f"  RMSE          : {rmse:.4f} m")
    print(f"  max error     : {max_err:.4f} m")

    stats = dict(n_samples=int(len(error)), mean_error_m=mean_err, std_error_m=std_err,
                 mae_m=mae, rmse_m=rmse, max_error_m=max_err,
                 note="error = 3D distance from ROV position to nearest point on the "
                      "pipeline centerline (dist_to_pipe_m), i.e. the standoff-tracking error")
    stats_path = os.path.join(EVAL_DIR, "tracking_error_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    save_pose_csvs(pipe_pts, traj, EVAL_DIR)

    # ---- figure: top-down profile, depth profile, error vs along-track ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # 1) top-down (x,y) profile
    ax = axes[0]
    ax.plot(pipe_xyz[:, 0], pipe_xyz[:, 1], "k-o", linewidth=2, markersize=4, label="Pipeline profile")
    ax.plot(traj["x"], traj["y"], "b-", linewidth=1.5, label="ROV profile")
    ax.scatter(traj["x"][0], traj["y"][0], c="green", s=60, zorder=5, label="Start")
    ax.scatter(traj["x"][-1], traj["y"][-1], c="red", s=60, marker="*", zorder=5, label="End")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top-down profile: pipeline vs ROV")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2) depth profile: z vs along-track distance for both, unrolled along the route
    ax = axes[1]
    ax.plot(pipe_s, pipe_xyz[:, 2], "k-o", linewidth=2, markersize=4, label="Pipeline profile")
    ax.plot(traj["along_track_m"], traj["z"], "b-", linewidth=1.5, label="ROV profile")
    ax.set_xlabel("along-track distance (m)")
    ax.set_ylabel("z (m, depth)")
    ax.set_title("Depth profile: pipeline vs ROV (unrolled along route)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3) tracking error vs time, with mean/RMSE reference lines
    ax = axes[2]
    ax.plot(traj["t_s"], error, color="tab:red", label="distance to pipe (error)")
    ax.axhline(rmse, color="tab:purple", linestyle="--", label=f"RMSE = {rmse:.3f} m")
    ax.axhline(mean_err, color="tab:orange", linestyle=":", label=f"mean = {mean_err:.3f} m")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error (m)")
    ax.set_title("Tracking error over the episode")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Pipeline profile vs ROV profile -- tracking error", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(EVAL_DIR, f"{out_prefix}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(EVAL_DIR, "eval_trajectory.csv"))
    parser.add_argument("--out-prefix", default="pipeline_vs_rov_profile")
    args = parser.parse_args()
    main(args.csv, args.out_prefix)
