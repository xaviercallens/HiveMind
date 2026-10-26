# QueenBee literature review: routing bandits, bee-inspired decision making, verified feedback, robust federated adapters, contribution incentives

Scope and method: alphaXiv searches on 2026-10-25 with seven queries (bandit routing of LLM queries; learned LLM routers; honeybee
dance and nest-site algorithms; Byzantine-robust and private federated learning; federated LoRA; execution feedback as preference
data; incentives for data and knowledge contribution). For each paper the title, venue line and abstract were read; **full texts
were not read**, so the notes below state what the abstracts claim and nothing finer. Ids are arXiv identifiers as listed by
alphaXiv (`https://www.alphaxiv.org/abs/<id>`). Papers from memory in the earlier draft were dropped unless the search returned
them.

## 1. Routing queries to experts as a bandit

| Paper | Id | What it does | What QueenBee takes from it |
|---|---|---|---|
| RouteLLM: Learning to Route LLMs with Preference Data | 2406.18665 | trains a router between a strong and a weak model from human preference data; cost-quality trade-off | the baseline shape (learned router, pairwise data) that a verified-outcome bandit must beat on our tasks |
| LLM Bandit: cost-efficient generation via preference-conditioned dynamic routing | 2502.02743 | treats model choice as a bandit conditioned on preferences | the bandit formulation; we replace preferences by verifier verdicts |
| Adaptive LLM Routing under Budget Constraints | 2508.21141 | online routing with a budget, bandit with cost | cost as part of the dance record, budget-aware recruitment |
| LLM Routing with Dueling Feedback | 2510.00841 | contextual dueling bandits: only pairwise feedback is available | the preference-pair strategy (b) is a dueling signal; verified pairs are its clean special case |
| Correlation-Aware Contextual Bandits with Surrogate Rewards for LLM Routing | 2607.09015 | correlated arms and surrogate rewards from an ML model | dance records are a surrogate reward; the cap and the lower bound guard against a biased surrogate |
| Online Learning with LLM Experts from Limited Feedback | 2609.05820 | bandit over K experts with limited feedback | the limited-feedback regime matches an IDE where the verifier runs only sometimes |
| ORACLE: orchestrator routing via adaptive verifier calibration feedback | 2607.22465 | routes in an agent pool using a verifier's calibrated feedback | closest to our setting: a verifier as the reward source; we add signed aggregate sharing across clients |
| OrcaRouter: hybrid offline-online learning for a production router | 2605.30736 | offline pretraining plus online updates | warm start from dances (offline) plus Thompson updates (online) is the same two-phase shape |
| The Routing Plateau: accuracy limits of LLM routers | 2606.07587 | shows router accuracy saturates | a warning: measure the plateau on our pool before claiming routing gains |
| LLMRouter: unified infrastructure for routers | 2608.06867 | a common evaluation harness | candidate harness for QB-1 with real models |
| Agent-as-a-Router: agentic routing for coding tasks | 2606.22902 | an agent decides which model to call for code | the Queen as planner-and-router; evaluate against it |
| AgentRouter / Planner-as-Router | 2609.22951, 2609.32917 | route each step of a multi-step workflow at plan time | the decomposition step (QB-2) has prior art; our addition is per-step verifiers and dance evidence |

Reading of the field: routing is well studied, learned routers plateau, and the newest work moves the reward from human
preference toward verifier feedback. QueenBee's bandit is not new; what is less common is a reward that is a test verdict, a
signed aggregate report shared between clients, and a formal cap on how far a report can move a router.

## 2. Bee-inspired collective decision making

| Paper | Id | What it does | Use |
|---|---|---|---|
| Multi-agent decision-making dynamics inspired by honeybees | 1711.11578 | value-sensitive nest-site model, near-equal options decided efficiently | the recruitment-proportional-to-quality rule and the stop signal map to weights and fading |
| Honeybee-like collective decision making in a kilobot swarm | 2310.15592 | hardware replication of nest-site selection | evidence the rule works with noisy, local reports |
| Consensus formation and relative stimulus perception in quality-sensitive agent systems | 2403.14856 | exploration plus social interaction, analysed | exploration floor in `probabilities(floor=...)` is the individual-exploration term |
| The Hive Mind is a Single Reinforcement Learning Agent | 2410.17517 | collective decision making as one RL agent | the view that the colony's rule is itself a bandit |
| Hive geometry shapes the recruitment rate of honeybee colonies | 2012.00157 | recruitment rate depends on where dances can be seen | a board that every router reads is the "dance floor"; its reach sets recruitment speed |
| Dancing honey bee robot elicits dance-following | 1803.07126 | a robot dance recruits real foragers | a reminder that a convincing dance recruits regardless of truth, hence signatures and caps |

## 3. Verified feedback as training signal (strategy b)

| Paper | Id | Note |
|---|---|---|
| Function-Level Execution Feedback for Code Preference Optimization | 2608.23632 | execution feedback per function as preference signal |
| ACE: adversarial unit test generation and preference optimisation | 2605.16299 | self-generated tests plus preference optimisation |
| ExecuCritic: calibrated critic shaping with verifiable rewards | 2609.16604 | unit tests as the objective reward; critic to shape it |
| Feedback Over Form: execution feedback matters more than pipeline topology (1-3B) | 2604.21950 | for small local models, feedback beats pipeline design |
| ZeroCoder: improving code generation without ground truth | 2604.07864 | RL with verifiable rewards without gold solutions |
| Personalized Federated DPO via LoRA adapters | 2610.12885 | preference optimisation across clients with adapters |

Reading: the repository already learned this lesson the hard way (NIGHT-MOE-1: gold-text fine-tuning hurt maths, see
`docs/NIGHT_MOE_RESULTS.md`); the literature agrees that execution feedback, not gold text, is the signal for code.

## 4. Robust, private aggregation of adapter updates (strategy c)

| Paper | Id | Note |
|---|---|---|
| Byzantine-Robust and Differentially Private Federated Optimization under Weaker Assumptions | 2603.23472 | robustness and privacy together |
| DP-BREM: private and Byzantine-robust FL with client momentum | 2306.12608 | the two defences combined with momentum |
| Differentially Private Learning with Adaptive Clipping | 1905.03871 | clip norm adapted online |
| An Experimental Study of Byzantine-Robust Aggregation Schemes | 2302.07173 | compares Krum-family rules empirically |
| Delayed Momentum Aggregation under partial participation | 2509.02970 | robustness when only some clients arrive, like our quorum |
| FedEx-LoRA: exact aggregation for federated LoRA | 2410.09432 | averaging LoRA factors is not averaging the product |
| CF-LoRA, CFLoRA, FedPA-LoRA, Gauge-aware server representations | 2609.36986, 2609.33147, 2608.15381, 2605.06733 | the factor-aggregation mismatch and fixes |
| From Bilinear to Linear: DP federated LoRA | 2609.26091 | privacy for the low-rank parameterisation |

Reading: averaging LoRA factors A and B separately is wrong in general (FedEx-LoRA and the 2026 follow-ups); our
`aggregate_shared_updates` averages whatever tensors it is given, so the caller must pass product-space deltas or use one of the
exact schemes. Clipping and noise are implemented; no (epsilon, delta) guarantee is claimed, in line with the adaptive-clipping
literature that treats the budget as a separate accounting step.

## 5. Incentives for contributed knowledge

| Paper | Id | Note |
|---|---|---|
| Collaborative ML with incentive-aware model rewards | 2010.12797 | reward contributors with model quality, not money |
| A Ratio-Based Shapley Value for Collaborative ML | 2510.13261 | contribution valuation that is incentive compatible |
| On the Volatility of Shapley-Based Contribution Metrics in FL | 2405.08044 | Shapley estimates are unstable; a warning against them as the only metric |
| Incentivizing High-quality Participation from FL Agents | 2506.16731 | quality, not volume, must be rewarded |
| Market Design for AI: Beyond the Copyright Binary | 2606.12260 | a market for human content used in training |
| FedMark-FM: auditable, risk-adjusted data markets for foundation-model adaptation | 2607.07529 | markets for adapters, prompts, preference data |
| Federated Learning for Data Market: Shapley-UCB | 2410.09107 | bandit over sellers in a data market |

Reading: contribution valuation by Shapley is attractive but volatile; a verified, capped point per admitted share is cruder but
auditable, and paying it as a lump sum keeps the existing Kelly equilibrium intact (Theorems 137-139). Whether points should later
be weighted by downstream use is an open question for a later season.

## 6. What this review changes in the proposal

1. The bandit is the default strategy (owner's preference, and the field's direction toward verifier rewards).
2. Pairs come only from verified corrections; gold text is not used.
3. Shared adapter deltas must be product-space or use an exact aggregation scheme; this is a stated limitation of strategy c.
4. Knowledge points are per admitted share, capped, lump-sum; Shapley weighting is deferred.
5. The routing plateau must be measured before any claim that routing beats the best single expert.
