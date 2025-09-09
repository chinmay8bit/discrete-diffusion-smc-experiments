#!/usr/bin/env bash
# Set DRY_RUN=true to only print commands (safer for a first check).

set -euo pipefail
# set -x   # Print each command before executing it

export CUDA_VISIBLE_DEVICES="6"

DRY_RUN=false   # set to false to actually run
PYTHON_CMD=python
SCRIPT=smc_scripts/smc_remdm_binarized_multi_runs.py
LOG_DIR=smc_mdm_script_logs
mkdir -p "$LOG_DIR"

# --- parameter lists (from your request) ---
num_particles=(2 4 8 16 32)
kl_weights=(1.0)
target_digits=(4)
masking_schedules=(linear)                    # you listed only "linear"
discretization_schedules=(cosine)
lambda_schedule_types=(linear)                # you listed "linear"
lambda_one_afters=(100)
reward_clamp_maxs=(-0.1)
phis=(100)
use_partial_options=(true)
perform_final_resample=true
runs_per_method=30

# optional extras (not requested -> left unset)
# partial_resample_size is left unset (None)
# ESS_min left unset

# compute total
total=0
for p in "${num_particles[@]}"; do
  for k in "${kl_weights[@]}"; do
    for t in "${target_digits[@]}"; do
      for m in "${masking_schedules[@]}"; do
        for d in "${discretization_schedules[@]}"; do
          for ltype in "${lambda_schedule_types[@]}"; do
            for la in "${lambda_one_afters[@]}"; do
              for rc in "${reward_clamp_maxs[@]}"; do
                for phi in "${phis[@]}"; do
                  for upr in "${use_partial_options[@]}"; do
                    total=$((total + 1))
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Will generate $total commands. DRY_RUN=$DRY_RUN"
echo

# run
cmd_index=0
for p in "${num_particles[@]}"; do
  for k in "${kl_weights[@]}"; do
    for t in "${target_digits[@]}"; do
      for m in "${masking_schedules[@]}"; do
        for d in "${discretization_schedules[@]}"; do
          for ltype in "${lambda_schedule_types[@]}"; do
            for la in "${lambda_one_afters[@]}"; do
              for rc in "${reward_clamp_maxs[@]}"; do
                for phi in "${phis[@]}"; do
                  for upr in "${use_partial_options[@]}"; do

                    (( ++cmd_index ))
                    # base args
                    args=()
                    args+=(--num_particles "$p")
                    args+=(--kl_weight "$k")
                    args+=(--target_digit "$t")
                    # you requested masking_schedule = linear (include explicitly)
                    args+=(--masking_schedule "$m")
                    args+=(--discretization_schedule "$d")
                    args+=(--lambda_schedule_type "$ltype")
                    args+=(--lambda_one_after "$la")
                    args+=(--reward_clamp_max "$rc")
                    args+=(--phi "$phi")
                    args+=(--runs_per_method "$runs_per_method")

                    # boolean flags
                    if [ "$upr" = "true" ]; then
                      args+=(--use_partial_resampling)
                      # if you later want to add --partial_resample_size, append here
                    fi

                    if [ "$perform_final_resample" = true ]; then
                      args+=(--perform_final_resample)
                    fi

                    # build command string for logging/printing
                    cmd=( "$PYTHON_CMD" "$SCRIPT" "${args[@]}" )
                    cmd_str=$(printf '%q ' "${cmd[@]}")

                    # create a log filename that identifies the run parameters
                    logname="${LOG_DIR}/run_p${p}_k${k}_t${t}_m${m}_d${d}_ltype${ltype}_la${la}_rc${rc}_phi${phi}_upr${upr}.log"

                    if [ "$DRY_RUN" = true ]; then
                      echo "[DRY] ($cmd_index/$total) $cmd_str"
                    else
                      echo "Running ($cmd_index/$total): $cmd_str"
                      # run and tee output to log
                      # you can change `nice` / `CUDA_VISIBLE_DEVICES` etc. here if needed
                      "${cmd[@]}" 2>&1 | tee "$logname"
                      # optional: small sleep to avoid totally hammering resources
                      sleep 0.2
                    fi

                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo
echo "Done. Generated $cmd_index commands."
if [ "$DRY_RUN" = true ]; then
  echo "Set DRY_RUN=false at top of the script to actually execute them."
fi
