# PMO-EvoRL

GPU-native PD-MORL MO-TD3-HER on EvoRL, JAX, and MuJoCo MJX.

```bash
conda activate pmo-evorl
export PYTHONPATH=.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Source-ratio baseline: 10 parallel environments.
python train.py

# Current conservative accelerated setting: 8.86x provisional throughput.
python train.py --num-envs 320

# Short numerical-stability run with a separate artifact directory.
python train.py --num-envs 320 --total-timesteps 320000 \
  --interpolator-eval-episodes 1 --eval-episodes 1 \
  --output-dir outputs/smoke_320

# Source-parity optimized variant: Lazy HER + sample_many.
python train.py --config configs/mo_td3_walker2d_source_optimized.yaml

# GPU-oriented variant: batch 1024, critic UTD 0.25.
python train.py --config configs/mo_td3_walker2d_gpu_optimized.yaml

# Use all three GPUs without changing one run's optimization batch:
# each GPU runs one GPU-native 320-environment training seed.
python run_multi_gpu.py --devices 0,1,2 --num-envs-per-gpu 320 \
  --total-timesteps 10000000 --output-root outputs/three_gpu

# Evaluate a completed checkpoint with native CPU MuJoCo.
python evaluate_cpu.py \
  --checkpoint outputs/three_gpu/seed_1/checkpoints/STEP \
  --output-dir outputs/three_gpu/seed_1/final_eval

# Synchronized medians at the production replay capacity.
python benchmark.py --num-envs 320 --mode full
python benchmark.py --num-envs 320 --mode env-only
python benchmark.py --num-envs 320 --mode learner-only
python benchmark.py --num-envs 320 --mode key-eval

# Stage-level profiling (microbenchmark; repeat for 10/80/160/320/640/1280 envs).
python benchmark_components.py --num-envs 640 --capacity 100000 \
  --profile-dir outputs/profiles/640
```

On a shared server, set `CUDA_VISIBLE_DEVICES` to the assigned GPU before running.
Training saves its final checkpoint before evaluation; full Pareto evaluation is an explicit offline step.
See [`specs/GPU_BENCHMARK.md`](specs/GPU_BENCHMARK.md) for the provisional results and limitations.
