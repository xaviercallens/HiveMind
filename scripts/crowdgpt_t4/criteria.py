"""Evaluate GOAL.md criteria 1-4 on results/p1 exactly as written. Writes results/p1/criteria.json."""
import json
import statistics as st

R = "/mnt/disks/disk-socrateai-local-1/crowdgpt_exp/results/p1/"


def g(n):
    return json.load(open(R + n + ".json"))


S = [g(f"sft_s{i}") for i in range(3)]
J = [g(f"jepa_s{i}") for i in range(3)]
sa, ja = [r["accuracy"] for r in S], [r["accuracy"] for r in J]
sf, jf = [r["far_loss"] for r in S], [r["far_loss"] for r in J]
pooled = ((st.variance(sa) + st.variance(ja)) / 2) ** 0.5
d = st.mean(ja) - st.mean(sa)
far_ok = st.mean(jf) <= st.mean(sf) + 0.05
L4 = g("jepa_s0_lam4")
out = {
    "sft_acc": sa, "jepa_acc": ja, "sft_mean": round(st.mean(sa), 3), "jepa_mean": round(st.mean(ja), 3),
    "sft_sd": round(st.stdev(sa), 3), "jepa_sd": round(st.stdev(ja), 3), "pooled_sd": round(pooled, 3),
    "diff_jepa_minus_sft": round(d, 3), "paired_diffs": [round(j - s, 2) for j, s in zip(ja, sa)],
    "sft_far": sf, "jepa_far": jf, "sft_far_mean": round(st.mean(sf), 4), "jepa_far_mean": round(st.mean(jf), 4),
    "far_diff": round(st.mean(jf) - st.mean(sf), 4), "base_far": 3.194,
    "c1_floor_pass": sa[0] >= 5.0,
    "c2_success": bool(d > 2 * pooled and far_ok),
    "c3_paper_like_gain": bool(d > 2 * pooled and far_ok and d >= 10),
    "c4_seeds_complete": len(sa) == 3 and len(ja) == 3,
    "secondary_prefix_acc": {"sft_s1_s2": [S[1].get("accuracy_prefix"), S[2].get("accuracy_prefix")],
                             "jepa_s1_s2": [J[1].get("accuracy_prefix"), J[2].get("accuracy_prefix")]},
    "exploratory_lam4_s0": {"acc": L4["accuracy"], "prefix": L4.get("accuracy_prefix"), "far": L4["far_loss"]},
    "jepa_term_last_logged": [r["log"][-1]["jepa"] for r in J],
    "train_seconds": {"sft": [r["train_seconds"] for r in S], "jepa": [r["train_seconds"] for r in J]},
}
json.dump(out, open(R + "criteria.json", "w"), indent=1)
print(json.dumps(out, indent=1))
