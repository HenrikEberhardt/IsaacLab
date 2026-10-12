#!/usr/bin/env bash
# Pick free GPU(s) on the shared workstation and print a system load summary.
#
# Usage: pick_gpu.sh [--need N]
# Prints FREE_GPU=<idx>[,<idx>...] and exits 0, or NO_FREE_GPU and exits 1.
#
# A GPU counts as free when it has no compute processes, uses less than
# GPU_MAX_MEM_MIB of memory and has utilization below GPU_MAX_UTIL percent.

set -euo pipefail

GPU_MAX_MEM_MIB="${GPU_MAX_MEM_MIB:-1000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
PREFERRED_GPU="${PREFERRED_GPU:-1}"
REPO_DIR="${REPO_DIR:-/hdd1/home/heberhardt/IsaacLab}"

need=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --need) need="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found"
    echo "NO_FREE_GPU"
    exit 1
fi

gpus=$(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)

echo "== GPUs (free = no processes, mem < ${GPU_MAX_MEM_MIB} MiB, util < ${GPU_MAX_UTIL} %) =="
printf "%-4s %-16s %-6s %-6s %s\n" "GPU" "MEM [MiB]" "UTIL" "STATE" "PROCESSES (user:pid:MiB)"

free=()
while IFS=', ' read -r idx uuid mem_used mem_total util; do
    procs=""
    nprocs=0
    while IFS=', ' read -r app_uuid pid app_mem; do
        [[ -z "${app_uuid:-}" || "$app_uuid" != "$uuid" ]] && continue
        user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        procs+="${user:-?}:${pid}:${app_mem} "
        nprocs=$((nprocs + 1))
    done <<< "$apps"

    state="busy"
    if (( nprocs == 0 && mem_used < GPU_MAX_MEM_MIB && util < GPU_MAX_UTIL )); then
        state="free"
        free+=("$idx")
    fi
    printf "%-4s %-16s %-6s %-6s %s\n" "$idx" "${mem_used}/${mem_total}" "${util}%" "$state" "${procs:--}"
done <<< "$gpus"

echo
echo "== CPU / RAM =="
echo "cores: $(nproc)   load average: $(cut -d' ' -f1-3 /proc/loadavg)"
free -h | head -2
echo
top -bn1 -o %CPU | sed -n '7,14p'
echo
echo "== Disk ($REPO_DIR) =="
df -h "$REPO_DIR" | tail -1
echo

# Put the preferred GPU first if it is free, keep the rest in index order.
ordered=()
for g in "${free[@]}"; do [[ "$g" == "$PREFERRED_GPU" ]] && ordered+=("$g"); done
for g in "${free[@]}"; do [[ "$g" != "$PREFERRED_GPU" ]] && ordered+=("$g"); done

if (( ${#ordered[@]} < need )); then
    echo "NO_FREE_GPU (need ${need}, free: ${#ordered[@]})"
    exit 1
fi

selected=$(IFS=,; echo "${ordered[*]:0:need}")
echo "FREE_GPU=${selected}"
