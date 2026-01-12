# AgentGuard

**AgentGuard** is a risk-aware evaluation and guarding framework for autonomous agents. It focuses on *trajectory-level risk assessment*, aiming to determine whether an agent’s execution trajectory contains safety risks under diverse application scenarios.

---

## 🔍 Overview

Autonomous agents (e.g., tool-using LLM agents, mobile agents, web agents) often execute multi-step trajectories consisting of observations, reasoning, and actions. Existing safety mechanisms mainly focus on **single-step content moderation** or **final-output filtering**, which are insufficient for capturing risks emerging *during execution*.

**AgentGuard** addresses this gap by:

* Modeling **agent trajectories** as first-class objects
* Providing **fine-grained risk taxonomy** across scenarios
* Supporting **trajectory-level risk detection and classification**
* Enabling both **offline evaluation** and **online guarding**

AgentGuard can be used as:

* A **benchmark** for agent safety evaluation
* A **risk classifier** for agent trajectories
* A **guard module** integrated into agent systems

---

## ✨ Key Features

* **Trajectory-level risk assessment** rather than single-turn analysis
* **Scenario-aware risk taxonomy** (task automation, web interaction, mobile control, etc.)
* **Multi-dimensional labels**: risk type, failure mode, severity
* **Model-agnostic**: works with different agent architectures and LLM backends
* **Extensible design** for new scenarios, risks, and guard policies

---

## 🧠 Methodology

### Problem Definition

Given an agent execution trajectory:

```
T = {(o₁, a₁), (o₂, a₂), ..., (oₙ, aₙ)}
```

AgentGuard aims to predict:

* Whether the trajectory is **risky**
* The **risk category** (e.g., security, privacy, physical harm)
* The **failure mode** (e.g., instruction hijacking, unsafe tool use)

### Core Idea

AgentGuard models risk as an emergent property of *sequential decisions* rather than isolated actions. It captures:

* Cross-step dependency
* Latent intent drift
* Compounded unsafe behaviors

### Architecture (Example)

* Trajectory encoder (text / multimodal)
* Risk-aware representation learning
* Multi-head prediction:

  * Risk label (safe / unsafe)
  * Risk type
  * Failure mode

> **Note**: The framework does not assume a specific backbone model; LLM-based, transformer-based, or hybrid encoders are all supported.

---

## 📊 Performance Highlights

> *(Fill in with your experimental results)*

* Evaluated on **AgentJudge**, **SAFE**, and internal benchmarks
* Outperforms step-level baselines in detecting:

  * Long-horizon instruction hijacking
  * Tool misuse after benign prefixes
* Strong generalization across:

  * Different agent frameworks
  * Different LLM backbones

Example metrics:

| Dataset    | Accuracy | F1 (Unsafe) | AUROC |
| ---------- | -------- | ----------- | ----- |
| AgentJudge | XX.X     | XX.X        | XX.X  |
| SAFE       | XX.X     | XX.X        | XX.X  |

---

## 🧩 Risk Taxonomy

AgentGuard uses a hierarchical risk taxonomy:

* **Risk Type**

  * Security & Privacy
  * Physical & Mental Wellbeing
  * Financial & Resource Abuse
  * System Misuse

* **Failure Mode** (examples)

  * Executed instructions from untrusted / embedded sources
  * Unsafe tool invocation
  * Over-privileged action chaining
  * Misaligned goal execution

This taxonomy covers risk scenarios overlooked by existing datasets.

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/your-org/AgentGuard.git
cd AgentGuard
pip install -r requirements.txt
```

### Basic Usage

```python
from agentguard import Guard

guard = Guard(model="agentguard-base")
result = guard.evaluate(trajectory)

print(result.is_unsafe)
print(result.risk_type)
print(result.failure_mode)
```

---

## 🧪 Evaluation

To reproduce experiments:

```bash
python scripts/evaluate.py \
  --dataset agentjudge \
  --model agentguard-base
```

Supported datasets:

* AgentJudge
* SAFE
* Custom JSON trajectories

---

## 🔌 Integration as an Online Guard

AgentGuard can be integrated into agent loops:

```python
for step in agent.run():
    guard_result = guard.check(step)
    if guard_result.block:
        break
```

Use cases:

* Real-time action blocking
* Risk-aware logging
* Human-in-the-loop escalation

---

## 📁 Repository Structure

```
AgentGuard/
├── agentguard/
│   ├── models/
│   ├── data/
│   ├── taxonomy/
│   └── guard.py
├── scripts/
│   ├── evaluate.py
│   └── preprocess.py
├── benchmarks/
├── README.md
└── requirements.txt
```

---

## 🛠️ Customization

* **Add new risk types**: `taxonomy/`
* **Support new agent formats**: implement a trajectory parser
* **Plug in new models**: follow `models/base.py`

---

## 📜 License

This project is released under the **Apache 2.0 License**.

---

## 📖 Citation

If you use AgentGuard in your research, please cite:

```bibtex
@article{,
  title={AgentGuard: Trajectory-Level Risk Assessment for Autonomous Agents},
  author={Anonymous},
  year={2025}
}
```

---

## 🤝 Acknowledgements

This project builds upon prior work in agent safety, trajectory evaluation, and risk-aware AI systems.
