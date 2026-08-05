#!/bin/bash
# Shared setup for every PBS job in filip_guide (ported from mini-embed-filip).
# Every job script starts with:
#
#   MASTER_PORT=29502
#   cd "${PBS_O_WORKDIR:?submit with qsub from the repo root}" || exit 1
#   source ./pbs_common.sh
#
# THE cd IS REQUIRED, AND IT HAS TO COME FIRST. PBS runs the job from a spooled
# copy with $HOME as the working directory, so `source ./pbs_common.sh` without
# the cd looks in $HOME and fails.
#
# config.py owns every model/dataset path — a job script must NOT export them, or
# the banner records something other than what the job read.
set -o pipefail

if [ ! -f config.py ] || [ ! -d src ]; then
    echo "FATAL: $(pwd) is not the filip_guide repo root (no config.py / src/)." >&2
    exit 1
fi

module load frameworks

# ---------------------------------------------------------------------------
# Python environment. MUST contain: torch (XPU/IPEX from `frameworks`),
# transformers>=4.57, mpi4py, AND the `esm` package which REGISTERS the `esmc`
# architecture into transformers (without it AutoModelForMaskedLM cannot load
# ESMC-300M). Create once on Aurora, e.g.:
#
#   module load frameworks
#   python -m venv --system-site-packages /flare/NLDesignProtein/bryan/envs/filip-guide-env
#   source /flare/NLDesignProtein/bryan/envs/filip-guide-env/bin/activate
#   pip install "esm @ git+https://github.com/Biohub/esm.git@main" transformers
#
# (--system-site-packages keeps the frameworks torch-xpu / oneCCL / IPEX build.)
# Override with FILIPG_ENV_ACTIVATE if your env lives elsewhere.
ENV_ACTIVATE=${FILIPG_ENV_ACTIVATE:-/flare/NLDesignProtein/bryan/envs/filip-guide-env/bin/activate}
source "${ENV_ACTIVATE}"

# --- topology: 12 tiles/node ---
RANKS_PER_NODE=${RANKS_PER_NODE:-12}
NNODES=$(wc -l < "$PBS_NODEFILE" | tr -d " ")
NRANKS=$(( NNODES * RANKS_PER_NODE ))

# --- tiles individually addressable + rendezvous for torch.distributed ---
# Set AFTER `module load frameworks`. The frameworks module defaults
# ONEAPI_DEVICE_SELECTOR to "opencl:gpu;level_zero:gpu", which enumerates each
# tile twice and breaks the one-rank-per-tile mapping; force Level-Zero only.
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# --- oneCCL / fabric ---
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=pidfd

# --pmi=pmix is REQUIRED to match CCL_PROCESS_LAUNCHER=pmix.
MPI_LAUNCH=(mpiexec -n "${NRANKS}" -ppn "${RANKS_PER_NODE}" --pmi=pmix --cpu-bind depth -d 8)

# ---------------------------------------------------------------------------
# job_banner "<title>" VAR1 VAR2 ...   (pass NAMES, not values)
# ---------------------------------------------------------------------------
job_banner() {
    local title="$1"; shift
    local dirty=""
    git diff --quiet 2>/dev/null || dirty=" (UNCOMMITTED CHANGES)"
    echo "================================================================"
    echo "  ${title}"
    echo "================================================================"
    echo "  job           : ${PBS_JOBID:-<interactive>}  ${PBS_JOBNAME:-}"
    echo "  started       : $(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "  user@host     : $(whoami)@$(hostname)"
    echo "  repo          : $(pwd)"
    echo "  git commit    : $(git rev-parse --short HEAD 2>/dev/null || echo '<no git>')${dirty}"
    echo "  env           : ${ENV_ACTIVATE}"
    echo "  topology      : ${NNODES} nodes x ${RANKS_PER_NODE} ranks/node = ${NRANKS} ranks"
    echo "----------------------------------------------------------------"
    echo "  device / fabric environment (effective, after module load):"
    local e
    for e in ONEAPI_DEVICE_SELECTOR ZE_FLAT_DEVICE_HIERARCHY OMP_NUM_THREADS \
             CCL_PROCESS_LAUNCHER CCL_ATL_TRANSPORT CCL_KVS_MODE FI_PROVIDER; do
        printf '    %-24s = %s\n' "$e" "${!e}"
    done
    echo "----------------------------------------------------------------"
    echo "  paths resolved by config.py:"
    python config.py 2>&1 | sed 's/^/    /'
    echo "----------------------------------------------------------------"
    echo "  checkpoints and hyperparameters set by this job script:"
    local v
    for v in "$@"; do
        printf '    %-24s = %s\n' "$v" "${!v}"
    done
    echo "================================================================"
    echo ""
}

# ---------------------------------------------------------------------------
# require_files VAR1 VAR2 ...  (variable NAMES holding paths; empty is skipped)
# ---------------------------------------------------------------------------
require_files() {
    local v missing=0
    for v in "$@"; do
        [ -z "${!v}" ] && continue
        if [ ! -e "${!v}" ]; then
            echo "FATAL: ${v}=${!v} does not exist" >&2
            missing=1
        fi
    done
    [ "$missing" = 1 ] && exit 1
    return 0
}

# ---------------------------------------------------------------------------
# finish <exit-code>   (optional Slack notify if ~/bin/slack_notify.sh exists)
# ---------------------------------------------------------------------------
finish() {
    local rc=$1
    if command -v slack >/dev/null 2>&1; then
        if [ "$rc" -eq 0 ]; then
            slack ":white_check_mark: DONE ${PBS_JOBID} ${PBS_JOBNAME}"
        else
            slack ":x: FAILED (exit ${rc}) ${PBS_JOBID} ${PBS_JOBNAME}"
        fi
    fi
    echo "[job] finished $(date '+%Y-%m-%dT%H:%M:%S%z') with exit code ${rc}"
    exit "$rc"
}

# Optional: Slack notifications. Harmless if absent (finish() guards on `slack`).
[ -f ~/bin/slack_notify.sh ] && source ~/bin/slack_notify.sh
