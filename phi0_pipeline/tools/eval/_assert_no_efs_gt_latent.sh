#!/usr/bin/env bash
# Abort if this shell is about to invoke the forbidden efs Dex3 gt_latent stack.
# Source from Phi_0_wpy eval launchers after setup_env.
_assert_no_efs_gt_latent() {
  local hit=""
  hit="$(
    { declare -p REPLAY_SH RUN_DIR VLA_REPLAY_SH GT_REPLAY_ROOT 2>/dev/null || true; } |
      grep -E 'sonic_latent_gt_replay/replay_gt_latent\.sh|/global_unittest/sonic_latent_gt_replay' ||
      true
  )"
  if [[ -n "${hit}" ]]; then
    echo "[guard] FORBIDDEN efs sonic_latent_gt_replay / replay_gt_latent.sh in env:" >&2
    echo "${hit}" >&2
    echo "[guard] use Phi_0_wpy tools/eval/launch_gt_sonic_replay.sh or run_sonic_latent_sim_eval.sh (subpackages + REVO2_HAND=1)" >&2
    return 1
  fi
  # Common accidental argv / exported script path
  if [[ "${1:-}" == *sonic_latent_gt_replay/replay_gt_latent.sh* ]]; then
    echo "[guard] FORBIDDEN: $1" >&2
    return 1
  fi
  return 0
}
_assert_no_efs_gt_latent "$@"
