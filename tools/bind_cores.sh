#!/bin/bash
# Find a free NUMA-local core block and print an mpirun prefix.
#
# Usage:
#   bash tools/bind_cores.sh <np>                    # auto-select best block
#   bash tools/bind_cores.sh <np> <start-end>        # validate a specific range
#   bash tools/bind_cores.sh <np> --taskset           # for scripts that spawn their own mpirun
#   $(bash tools/bind_cores.sh 24) python run/optimize.py --config ...
#   $(bash tools/bind_cores.sh 8 --taskset) bash run.sh 8   # SU2 native scripts
#
# Default output (mpirun mode):
#   mpirun -np 24 --cpu-set 0,1,2,...,23 --bind-to core
# With --taskset (for SU2 shape_optimization.py which spawns its own mpirun):
#   taskset -c 0,1,2,...,7
#
# Exit codes:
#   0  success (command printed to stdout)
#   1  not enough free cores / range validation failed

set -euo pipefail

MODE="mpirun"  # default: emit mpirun prefix
REQUESTED_RANGE=""
args=()
for arg in "$@"; do
    case "$arg" in
        --taskset) MODE="taskset" ;;  # for SU2 native scripts (they spawn their own mpirun)
        *) args+=("$arg") ;;
    esac
done

if [ ${#args[@]} -lt 1 ] || ! [[ "${args[0]}" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash tools/bind_cores.sh <np> [start-end] [--taskset]" >&2
    echo "  --taskset  emit 'taskset -c <cores>' instead of 'mpirun --cpu-set'" >&2
    echo "             (for scripts like SU2 shape_optimization.py that spawn their own mpirun)" >&2
    exit 1
fi

NP=${args[0]}
if [ ${#args[@]} -ge 2 ]; then
    if ! [[ "${args[1]}" =~ ^[0-9]+-[0-9]+$ ]]; then
        echo "ERROR: range must be start-end (e.g. 30-54)" >&2
        exit 1
    fi
    REQUESTED_RANGE="${args[1]}"
fi
SAMPLE_SECS=2
IDLE_THRESHOLD=10  # cores with <10% usage are "free"

# --- Discover NUMA topology ---------------------------------------------------

if ! command -v numactl &>/dev/null; then
    echo "ERROR: numactl not found. Install with: sudo apt install numactl" >&2
    exit 1
fi

# Filter out SMT siblings — keep only physical cores.
# /sys/devices/system/cpu/cpuN/topology/thread_siblings_list gives e.g. "0,128"
# for a hyperthread pair. We keep only the lowest-numbered sibling.
declare -A PHYSICAL_CORES
for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*/; do
    cpu_id=$(basename "$cpu_dir" | sed 's/cpu//')
    siblings_file="$cpu_dir/topology/thread_siblings_list"
    if [ -f "$siblings_file" ]; then
        first=$(tr ',' '\n' < "$siblings_file" | sed 's/-.*//; s/^0*//' | sort -n | head -1)
        first=${first:-0}
        if [ "$cpu_id" -eq "$first" ]; then
            PHYSICAL_CORES[$cpu_id]=1
        fi
    else
        PHYSICAL_CORES[$cpu_id]=1
    fi
done

echo "Physical cores: ${#PHYSICAL_CORES[@]} (hyperthreads excluded)" >&2

declare -A NUMA_CORES
while IFS= read -r line; do
    node=$(echo "$line" | awk '{print $2}')
    all_cores=$(echo "$line" | sed 's/^.*cpus: //')
    phys_only=""
    for c in $all_cores; do
        if [[ -v PHYSICAL_CORES[$c] ]]; then
            phys_only+="$c "
        fi
    done
    NUMA_CORES[$node]="${phys_only% }"
done < <(numactl --hardware | grep "^node [0-9]* cpus:")

NUM_NODES=${#NUMA_CORES[@]}
echo "NUMA nodes: $NUM_NODES" >&2
for n in $(echo "${!NUMA_CORES[@]}" | tr ' ' '\n' | sort -n); do
    count=$(echo "${NUMA_CORES[$n]}" | wc -w)
    echo "  node $n: $count cores [${NUMA_CORES[$n]}]" >&2
done

# --- Sample CPU usage ---------------------------------------------------------

echo "Sampling CPU usage for ${SAMPLE_SECS}s..." >&2

declare -A CORE_IDLE
while IFS= read -r line; do
    core=$(echo "$line" | awk '{print $2}')
    idle=$(echo "$line" | awk '{print $NF}')
    CORE_IDLE[$core]="$idle"
done < <(mpstat -P ALL "$SAMPLE_SECS" 1 | awk 'NR>3 && $2 ~ /^[0-9]+$/ {print}')

# --- Validate requested range, or auto-select ---------------------------------

if [ -n "$REQUESTED_RANGE" ]; then
    range_start=${REQUESTED_RANGE%-*}
    range_end=${REQUESTED_RANGE#*-}
    range_count=$(( range_end - range_start + 1 ))

    if [ "$range_end" -le "$range_start" ]; then
        echo "ERROR: end must be greater than start ($REQUESTED_RANGE)" >&2
        exit 1
    fi
    if [ "$range_count" -lt "$NP" ]; then
        echo "ERROR: range $REQUESTED_RANGE has $range_count cores but np=$NP" >&2
        exit 1
    fi

    # Check: all cores are physical
    non_physical=()
    for (( c=range_start; c<=range_end; c++ )); do
        if ! [[ -v PHYSICAL_CORES[$c] ]]; then
            non_physical+=("$c")
        fi
    done
    if [ ${#non_physical[@]} -gt 0 ]; then
        echo "ERROR: range includes hyperthreads (not physical cores): ${non_physical[*]}" >&2
        exit 1
    fi

    # Check: all cores on the same NUMA node
    range_node=""
    for n in $(echo "${!NUMA_CORES[@]}" | tr ' ' '\n' | sort -n); do
        match=true
        for (( c=range_start; c<=range_end; c++ )); do
            if ! echo " ${NUMA_CORES[$n]} " | grep -q " $c "; then
                match=false
                break
            fi
        done
        if $match; then
            range_node=$n
            break
        fi
    done
    if [ -z "$range_node" ]; then
        echo "ERROR: range $REQUESTED_RANGE spans multiple NUMA nodes" >&2
        for n in $(echo "${!NUMA_CORES[@]}" | tr ' ' '\n' | sort -n); do
            echo "  node $n: ${NUMA_CORES[$n]}" >&2
        done
        exit 1
    fi

    # Check: cores are free
    echo "Cores $range_start-$range_end on NUMA node $range_node" >&2
    busy=()
    for (( c=range_start; c<=range_end; c++ )); do
        idle=${CORE_IDLE[$c]:-100}
        usage=$(awk "BEGIN {printf \"%.0f\", 100 - $idle}")
        if [ "$usage" -ge "$IDLE_THRESHOLD" ]; then
            busy+=("$c(${usage}%)")
        fi
    done
    if [ ${#busy[@]} -gt 0 ]; then
        echo "WARNING: busy cores in range: ${busy[*]}" >&2
    else
        echo "All cores in range are idle" >&2
    fi

    # Build the cpu-set from the first NP cores in the range
    selected=()
    for (( c=range_start; c<=range_end && ${#selected[@]}<NP; c++ )); do
        selected+=("$c")
    done
    cpu_set=$(IFS=,; echo "${selected[*]}")

    echo "Validated: node $range_node, cores $cpu_set" >&2
    if [ "$MODE" = "taskset" ]; then
        echo "taskset -c $cpu_set"
    else
        echo "mpirun -np $NP --cpu-set $cpu_set --bind-to core"
    fi
    exit 0
fi

# --- Auto-select: find the best NUMA node -------------------------------------
# "Best" = the node with the most free cores (≥ NP). Among ties, pick the one
# with the highest average idle % (least contention on remaining cores).

best_node=-1
best_free=0
best_avg_idle=0
declare -A NODE_FREE_CORES

for n in $(echo "${!NUMA_CORES[@]}" | tr ' ' '\n' | sort -n); do
    free_list=()
    for c in ${NUMA_CORES[$n]}; do
        idle=${CORE_IDLE[$c]:-100}
        usage=$(awk "BEGIN {printf \"%.0f\", 100 - $idle}")
        if [ "$usage" -lt "$IDLE_THRESHOLD" ]; then
            free_list+=("$c")
        fi
    done

    free_count=${#free_list[@]}
    NODE_FREE_CORES[$n]="${free_list[*]}"
    echo "  node $n: $free_count free cores" >&2

    if [ "$free_count" -ge "$NP" ]; then
        avg_idle=0
        for c in "${free_list[@]}"; do
            avg_idle=$(awk "BEGIN {printf \"%.0f\", $avg_idle + ${CORE_IDLE[$c]:-100}}")
        done
        avg_idle=$((avg_idle / free_count))

        if [ "$free_count" -gt "$best_free" ] || \
           { [ "$free_count" -eq "$best_free" ] && [ "$avg_idle" -gt "$best_avg_idle" ]; }; then
            best_node=$n
            best_free=$free_count
            best_avg_idle=$avg_idle
        fi
    fi
done

if [ "$best_node" -lt 0 ]; then
    echo "ERROR: no single NUMA node has $NP free cores." >&2
    echo "Free cores per node:" >&2
    for n in $(echo "${!NUMA_CORES[@]}" | tr ' ' '\n' | sort -n); do
        echo "  node $n: $(echo "${NODE_FREE_CORES[$n]}" | wc -w) free" >&2
    done
    echo "Reduce np or free up cores." >&2
    exit 1
fi

# Pick the first NP free cores from the best node (they're already sorted)
selected=(${NODE_FREE_CORES[$best_node]})
selected=("${selected[@]:0:$NP}")
cpu_set=$(IFS=,; echo "${selected[*]}")

echo "Selected: node $best_node, cores $cpu_set" >&2

# --- Print the command (stdout only) ------------------------------------------

if [ "$MODE" = "taskset" ]; then
    echo "taskset -c $cpu_set"
else
    echo "mpirun -np $NP --cpu-set $cpu_set --bind-to core"
fi
