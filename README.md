# README: Using Isaac Sim for the ROV Reinforcement Learning Digital Twin

This guide describes how to integrate **NVIDIA Isaac Sim** into an existing **ROV reinforcement learning (RL) digital twin**, combining a **6-DOF Fossen hydrodynamic model**, **physics-informed neural network (PINN)** components, and a **PPO reinforcement learning agent**. The workflow is intended for deployment and experimentation on an **NVIDIA DGX Spark with ARM64/Grace architecture**.

## 1. Project Overview

The objective is to create a simulation and reinforcement learning pipeline in which the ROV digital twin provides a physically meaningful environment for training and evaluating autonomous control policies.

The overall workflow is:

```text
Fossen Hydrodynamic Model
          │
          ▼
   PINN-Based Physics
          │
          ▼
     ROV Digital Twin
          │
          ▼
      NVIDIA Isaac Sim
          │
          ▼
   RL Environment / API
          │
          ▼
       PPO Agent
          │
          ▼
   Trained ROV Policy
          │
          ▼
 Isaac Sim Validation
```

The existing Fossen model provides the underlying marine dynamics, while Isaac Sim provides the simulation environment, physics, sensors, visualization, and deployment framework for the RL workflow.

---

# 2. Hardware and Software Environment

The recommended development platform is:

* **Hardware:** NVIDIA DGX Spark
* **Architecture:** ARM64 / NVIDIA Grace
* **GPU:** NVIDIA Blackwell-class GPU
* **Simulation:** NVIDIA Isaac Sim
* **RL Algorithm:** Proximal Policy Optimization (PPO)
* **ROV Dynamics:** 6-DOF Fossen model
* **Physics Learning:** PINN-based modelling
* **Container Runtime:** Docker
* **GPU Runtime:** NVIDIA Container Toolkit
* **Asset Format:** USD
* **Programming:** Python

The project directory can be organised as:

```text
~/rov-isaac/
│
├── assets/
│   ├── rov/
│   ├── environment/
│   └── sensors/
│
├── usd/
│   ├── rov.usd
│   └── underwater_scene.usd
│
├── models/
│   ├── fossen/
│   ├── pinn/
│   └── ppo/
│
├── scripts/
│   ├── environment/
│   ├── dynamics/
│   ├── training/
│   └── evaluation/
│
├── checkpoints/
│
├── logs/
│
└── README.md
```

---

# 3. Environment Setup on DGX Spark

Isaac Sim releases and container availability can differ between **x86_64** and **ARM64/Grace** platforms. Therefore, the DGX Spark should be configured using the appropriate NVIDIA-supported ARM64 container rather than assuming that the standard x86_64 installation will work.

Before starting, verify:

1. NVIDIA GPU driver
2. Docker
3. NVIDIA Container Toolkit
4. ARM64-compatible Isaac Sim container
5. GPU passthrough
6. Headless rendering

Check the architecture:

```bash
uname -m
```

The expected result is:

```text
aarch64
```

Check the NVIDIA driver:

```bash
nvidia-smi
```

Check Docker:

```bash
docker --version
```

Check that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

The CUDA image tag should be adjusted if your installed driver requires a different compatible CUDA version.

---

# 4. Obtain the Isaac Sim Container

Isaac Sim is distributed by NVIDIA through the **NVIDIA NGC container registry**.

For an ARM64/Grace system, use the appropriate **aarch64** image tag provided by NVIDIA.

For example:

```bash
docker pull nvcr.io/nvidia/isaac-sim:<tag>-aarch64
```

Replace `<tag>` with the Isaac Sim version required for the project.

For example, if using a specific supported release:

```bash
docker pull nvcr.io/nvidia/isaac-sim:<ISAAC_SIM_VERSION>-aarch64
```

> **Important:** Do not assume that an x86_64 Isaac Sim container will run correctly on DGX Spark. Confirm that the selected release explicitly supports the ARM64/Grace platform.

---

# 5. Create the ROV Isaac Sim Workspace

Create a dedicated project directory:

```bash
mkdir -p ~/rov-isaac
```

Create the main subdirectories:

```bash
mkdir -p ~/rov-isaac/{assets,usd,models,scripts,checkpoints,logs}
```

The directory can then be mounted into the Isaac Sim container.

---

# 6. Launch Isaac Sim

Launch the container with GPU access and mount the ROV project directory:

```bash
docker run --gpus all -it --rm \
    -v ~/rov-isaac:/workspace \
    nvcr.io/nvidia/isaac-sim:<tag>-aarch64
```

Inside the container, verify the mounted directory:

```bash
ls -la /workspace
```

You should see:

```text
assets
usd
models
scripts
checkpoints
logs
```

---

# 7. Verify Isaac Sim

Before developing the ROV digital twin, verify that Isaac Sim itself can start successfully.

For a headless DGX Spark training environment:

```bash
./isaac-sim.headless.sh
```

If the installation uses a different launcher or installation layout, use the corresponding Isaac Sim executable supplied by the container.

The first objective is simply to confirm:

```text
GPU detected
       ↓
Isaac Sim starts
       ↓
Renderer initialises
       ↓
Headless mode works
```

Do this **before** adding the ROV model, Fossen dynamics, PINN or PPO components.

---

# 8. ROV Digital Twin Architecture

The proposed digital twin consists of several interacting layers:

```text
                 ┌──────────────────────┐
                 │     PPO Agent        │
                 │ Policy / Value Net    │
                 └──────────┬───────────┘
                            │
                       Actions
                            │
                            ▼
                 ┌──────────────────────┐
                 │   ROV Controller     │
                 │ Thruster Allocation  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Isaac Sim ROV      │
                 │  USD + Physics       │
                 │ Sensors + Environment│
                 └──────────┬───────────┘
                            │
                       State Data
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Fossen 6-DOF Model   │
                 │ Hydrodynamics        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ PINN / Physics Model │
                 │ Parameter Estimation │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Observation / Reward │
                 │      Function        │
                 └──────────┬───────────┘
                            │
                            └──────► PPO
```

---

# 9. ROV Model Integration

The ROV should be represented in Isaac Sim using a USD-based asset.

Recommended components include:

* ROV body
* Six-degree-of-freedom rigid-body model
* Thrusters
* Thruster forces
* IMU
* Depth sensor
* Camera
* Sonar or simulated range sensor
* Navigation state
* Pipeline/environment geometry

The ROV state can be represented as:

```text
η = [x, y, z, φ, θ, ψ]

ν = [u, v, w, p, q, r]
```

where:

* `x, y, z` = position
* `φ, θ, ψ` = roll, pitch and yaw
* `u, v, w` = linear velocities
* `p, q, r` = angular velocities

Together:

```text
x_ROV = [η, ν]
```

---

# 10. Fossen Hydrodynamic Model

The Fossen model provides the marine vehicle dynamics used by the digital twin.

A general 6-DOF representation is:

```text
Mν̇ + C(ν)ν + D(ν)ν + g(η) = τ + τdist
```

where:

* `M` = rigid-body and added-mass matrix
* `C(ν)` = Coriolis and centripetal matrix
* `D(ν)` = hydrodynamic damping
* `g(η)` = restoring forces and moments
* `τ` = control input
* `τdist` = environmental disturbances

The model can be used as a reference physics model while Isaac Sim handles the simulated environment and sensor interface.

---

# 11. PINN-Based Physics Model

The PINN component can be used to learn or refine uncertain hydrodynamic parameters.

Potential applications include:

* Hydrodynamic coefficient estimation
* Added-mass estimation
* Drag estimation
* Disturbance modelling
* Model correction
* Sim-to-real adaptation

A conceptual structure is:

```text
Experimental / Simulated Data
             │
             ▼
       Neural Network
             │
             ▼
   Physics-Based Loss
             │
             ├── Data Loss
             ├── Dynamics Loss
             └── Boundary / Constraint Loss
             │
             ▼
      PINN Parameters
```

The PINN should complement the Fossen model rather than completely replacing the established dynamics model unless that is specifically required by the experiment.

---

# 12. PPO Reinforcement Learning

The PPO agent receives observations from the ROV environment and produces control actions.

Example observation vector:

```text
o_t =
[
    position,
    orientation,
    linear_velocity,
    angular_velocity,
    tracking_error,
    heading_error,
    depth_error
]
```

The action vector may represent desired thruster commands:

```text
a_t =
[
    T1,
    T2,
    T3,
    T4,
    T5,
    T6
]
```

The exact number of thrusters should match the physical ROV configuration.

A typical training loop is:

```text
Reset Environment
       │
       ▼
Obtain ROV Observation
       │
       ▼
PPO Policy
       │
       ▼
Thruster Commands
       │
       ▼
Isaac Sim
       │
       ▼
ROV State Update
       │
       ▼
Calculate Reward
       │
       ▼
Next Observation
       │
       └──────────────► PPO Update
```

---

# 13. Pipeline-Tracking Task

For a pipeline-tracking experiment, the environment should contain:

* Underwater terrain
* Pipeline geometry
* ROV
* Navigation reference
* Disturbance model
* Sensors
* Tracking reward

A simplified tracking error can be defined as:

```text
e = p_ROV - p_reference
```

The reward can combine several objectives:

```text
R =
-w1 ||e_position||
-w2 ||e_heading||
-w3 ||e_velocity||
-w4 ||u||
```

where the weights should be selected experimentally.

The objective is to minimise tracking error while avoiding excessive control effort and unstable manoeuvres.

---

# 14. Training and Evaluation

Training should initially be performed in a simplified environment.

### Stage 1: Controller verification

```text
ROV
 ↓
Fossen model
 ↓
Basic controller
 ↓
Tracking test
```

### Stage 2: Isaac Sim integration

```text
ROV USD
 ↓
Isaac Sim
 ↓
Sensors
 ↓
Fossen dynamics
```

### Stage 3: PPO training

```text
Isaac Sim
 ↓
RL Environment
 ↓
PPO
 ↓
Policy
```

### Stage 4: Disturbance training

Introduce:

* Ocean currents
* Sensor noise
* Parameter uncertainty
* Thruster uncertainty
* External disturbances

### Stage 5: Validation

Evaluate the trained policy under conditions that were **not used during training**.

---

# 15. Recommended Project Workflow

The complete workflow is:

```text
1. Configure DGX Spark
          ↓
2. Verify NVIDIA driver
          ↓
3. Verify Docker + GPU passthrough
          ↓
4. Pull ARM64 Isaac Sim container
          ↓
5. Verify headless Isaac Sim
          ↓
6. Import ROV USD model
          ↓
7. Configure sensors
          ↓
8. Integrate Fossen dynamics
          ↓
9. Add PINN model
          ↓
10. Create RL environment
          ↓
11. Implement reward function
          ↓
12. Connect PPO agent
          ↓
13. Train policy
          ↓
14. Evaluate tracking performance
          ↓
15. Test robustness
          ↓
16. Validate digital twin
```

---

# 16. Troubleshooting

### Check GPU visibility

```bash
nvidia-smi
```

If the GPU is not visible inside the container, check the NVIDIA Container Toolkit and Docker GPU runtime configuration.

### Check container architecture

```bash
uname -m
```

Expected on DGX Spark:

```text
aarch64
```

### Check mounted workspace

```bash
ls -la /workspace
```

### Test Isaac Sim headless mode

```bash
./isaac-sim.headless.sh
```

If headless startup fails, resolve the Isaac Sim/container/driver issue before attempting to run the RL training pipeline.

### Check Python environment

```bash
python --version
```

and:

```bash
python -c "import torch; print(torch.__version__)"
```

---

# 17. Reproducibility

Record the following information for every experiment:

```text
Isaac Sim version:
Container image:
DGX Spark configuration:
NVIDIA driver:
CUDA version:
Python version:
PyTorch version:
RL framework:
PPO configuration:
Fossen model parameters:
PINN configuration:
ROV configuration:
Simulation timestep:
Training timestep:
Random seed:
Number of training episodes:
```

This information is particularly important when comparing different PPO policies or transferring policies between simulation configurations.

---

# 18. Final Objective

The final system should provide a unified **ROV reinforcement learning digital twin** in which:

```text
        Fossen Model
             +
        PINN Physics
             +
        Isaac Sim
             +
       ROV Sensors
             +
          PPO RL
             │
             ▼
   Autonomous ROV Controller
             │
             ▼
     Pipeline Tracking
             │
             ▼
   Robustness / Validation
```

The DGX Spark provides the computational platform for running the simulation, neural-network training and evaluation workflow. The resulting environment can then serve as a foundation for further work on **sim-to-real transfer, autonomous underwater navigation, adaptive control, and agentic AI for subsea robotics**.
