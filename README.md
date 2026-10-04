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

# Compare steady-state training throughput at the same update ratios.
python benchmark.py --num-envs 10 --steps 10
python benchmark.py --num-envs 320 --steps 10
```

On a shared server, set `CUDA_VISIBLE_DEVICES` to the assigned GPU before running.
See [`specs/GPU_BENCHMARK.md`](specs/GPU_BENCHMARK.md) for the provisional results and limitations.
