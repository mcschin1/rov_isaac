# Using Isaac Sim for the ROV Reinforcement Learning Digital Twin

This project explores the integration of **NVIDIA Isaac Sim** with an existing **ROV reinforcement learning (RL) digital twin**. The framework combines marine vehicle dynamics, physics-informed learning and reinforcement learning to investigate autonomous ROV control and pipeline-tracking applications.

The implementation is designed for GPU-accelerated simulation and training on NVIDIA hardware.

> **Note:** This README provides an overview of the workflow and environment setup. Certain implementation details, model parameters, training configurations, simulation assets and integration interfaces have been intentionally omitted.

---

## 1. System Overview

The digital twin combines several components:

```text
        ROV Hydrodynamic Model
                 │
                 ▼
        Physics-Informed Model
                 │
                 ▼
          ROV Digital Twin
                 │
                 ▼
           NVIDIA Isaac Sim
                 │
                 ▼
          RL Environment
                 │
                 ▼
             PPO Agent
                 │
                 ▼
       Autonomous ROV Control
```

Isaac Sim provides the simulation environment, while the existing marine dynamics and learning components provide the basis for ROV behaviour and control.

---

## 2. Hardware and Software

The development workflow uses:

* NVIDIA GPU computing platform
* ARM64/Grace-based computing environment
* NVIDIA Isaac Sim
* Docker-based deployment
* Python
* PPO reinforcement learning
* 6-DOF marine vehicle dynamics
* Physics-informed neural networks (PINNs)
* USD-based simulation assets

Specific hardware configurations, software versions and dependency combinations are intentionally not disclosed in this public README.

---

## 3. Isaac Sim Environment

Isaac Sim is deployed using an NVIDIA-supported container appropriate for the target GPU architecture.

A generic container workflow is:

```bash
docker pull nvcr.io/nvidia/isaac-sim:<VERSION>-<ARCH>
```

The container is then launched with GPU access and a project workspace:

```bash
docker run --gpus all -it --rm \
    -v <PROJECT_DIRECTORY>:/workspace \
    nvcr.io/nvidia/isaac-sim:<VERSION>-<ARCH>
```

The exact image version, architecture tag and project directory used in the implementation are intentionally omitted.

---

## 4. Initial Verification

Before integrating the ROV digital twin, Isaac Sim should be verified independently.

For a headless environment, the corresponding Isaac Sim headless launcher can be used:

```bash
./isaac-sim.headless.sh
```

The initial verification should confirm:

```text
GPU
 │
 ▼
Container
 │
 ▼
Isaac Sim
 │
 ▼
Headless Rendering
 │
 ▼
Simulation
```

Only after this stage has been successfully completed should the ROV environment be introduced.

---

## 5. ROV Digital Twin

The ROV digital twin incorporates a six-degree-of-freedom marine vehicle representation.

The general state consists of position, orientation, linear velocity and angular velocity:

```text
ROV State
   │
   ├── Position
   ├── Orientation
   ├── Linear Velocity
   └── Angular Velocity
```

The underlying hydrodynamic formulation follows established marine vehicle modelling approaches.

Specific hydrodynamic coefficients, vehicle dimensions, mass properties, damping parameters and other calibrated model parameters are not included.

---

## 6. Physics-Informed Learning

A physics-informed neural network is incorporated to represent selected aspects of the ROV's uncertain or nonlinear behaviour.

The general concept is:

```text
Simulation / Experimental Data
             │
             ▼
       Neural Network
             │
             ▼
      Physics Constraints
             │
             ▼
     Physics-Informed Model
```

The PINN can support applications such as:

* Hydrodynamic parameter estimation
* Model correction
* Uncertainty representation
* Disturbance modelling
* Adaptive simulation

The network architecture, training parameters, loss-function formulation and calibrated parameters are implementation-specific and are therefore omitted.

---

## 7. Reinforcement Learning

The RL component uses PPO to learn an ROV control policy.

The general training loop is:

```text
ROV Observation
       │
       ▼
    PPO Policy
       │
       ▼
 Control Action
       │
       ▼
   Isaac Sim
       │
       ▼
   New State
       │
       ▼
    Reward
       │
       └──────────► PPO Update
```

The policy is trained to achieve the required ROV control objective while maintaining stable and physically meaningful behaviour.

Exact observation definitions, action mappings, network architecture, PPO hyperparameters and reward coefficients are not disclosed.

---

## 8. Pipeline-Tracking Application

One application considered by the digital twin is autonomous underwater pipeline tracking.

The general objective is:

```text
Reference Pipeline
        │
        ▼
Tracking Error
        │
        ▼
   PPO Controller
        │
        ▼
    ROV Motion
        │
        ▼
Tracking Performance
```

The simulation can incorporate environmental disturbances and sensor uncertainty to evaluate the robustness of the learned controller.

The precise pipeline geometry, sensor configuration, tracking thresholds and reward formulation are implementation-specific.

---

## 9. Development Workflow

The recommended development sequence is:

```text
1. Configure GPU environment
          ↓
2. Verify container GPU access
          ↓
3. Start Isaac Sim
          ↓
4. Verify headless simulation
          ↓
5. Import ROV simulation asset
          ↓
6. Configure sensors
          ↓
7. Connect vehicle dynamics
          ↓
8. Integrate physics-informed model
          ↓
9. Create RL environment
          ↓
10. Configure PPO training
          ↓
11. Train controller
          ↓
12. Evaluate ROV performance
```

---

## 10. Research Objective

The overall objective is to establish a GPU-accelerated digital twin framework for investigating:

* Autonomous ROV control
* Reinforcement learning
* Physics-informed learning
* Pipeline tracking
* Robust control under uncertainty
* Sim-to-real transfer
* Intelligent subsea robotics

The public version of this repository provides the **conceptual workflow and environment setup**, while the complete implementation remains restricted to the appropriate research and development environment.

This version gives readers enough information to understand **what you have built and how the components interact**, while making it substantially harder to reproduce the complete system without your internal code, parameters and assets.
