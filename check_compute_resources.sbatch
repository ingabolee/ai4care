#!/bin/bash
#SBATCH --job-name=ai4care-resource-check
#SBATCH --partition=dl-gpu-a5000
#SBATCH --gpus=1
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=compute-resources-%j.out
#SBATCH --error=compute-resources-%j.err

set -u

echo "=== Slurm compute-node resource check ==="
echo "Time       : $(date --iso-8601=seconds)"
echo "Hostname   : $(hostname)"
echo "Node       : ${SLURMD_NODENAME:-unknown}"
echo "Job ID     : ${SLURM_JOB_ID:-unknown}"
echo "Partition  : ${SLURM_JOB_PARTITION:-unknown}"
echo

echo "=== Slurm allocation ==="
for name in SLURM_JOB_CPUS_PER_NODE SLURM_CPUS_ON_NODE SLURM_CPUS_PER_TASK \
            SLURM_NTASKS SLURM_NNODES SLURM_JOB_NODELIST SLURM_JOB_GPUS \
            CUDA_VISIBLE_DEVICES; do
  value="$(printenv "$name" 2>/dev/null || true)"
  printf '%-28s %s\n' "$name" "${value:-not set}"
done
echo

echo "=== CPU hardware and job affinity ==="
if command -v lscpu >/dev/null 2>&1; then
  lscpu
else
  echo "lscpu is unavailable"
fi
echo "Online CPUs visible to this process: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status 2>/dev/null || true
echo

echo "=== RAM ==="
if command -v free >/dev/null 2>&1; then
  free -h
fi
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo 2>/dev/null || true
for memory_file in /sys/fs/cgroup/memory.max \
                   /sys/fs/cgroup/memory.current \
                   /sys/fs/cgroup/memory/memory.limit_in_bytes \
                   /sys/fs/cgroup/memory/memory.usage_in_bytes; do
  if [[ -r "$memory_file" ]]; then
    printf '%s: ' "$memory_file"
    cat "$memory_file"
  fi
done
echo

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
    --format=csv,noheader 2>&1 || true
  nvidia-smi 2>&1 || true
else
  echo "nvidia-smi is unavailable on this compute node."
fi
echo

echo "=== Slurm node record (if permitted) ==="
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURMD_NODENAME:-}" ]]; then
  scontrol show node "$SLURMD_NODENAME" 2>&1 || true
else
  echo "scontrol or the compute-node name is unavailable."
fi

echo
echo "Resource check completed on $(hostname) at $(date --iso-8601=seconds)."
