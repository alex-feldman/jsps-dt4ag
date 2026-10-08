#!/usr/bin/env bash
# Train one arm of a mask-variant comparison on a COLMAP model shared with a donor run.
#
# Copies the donor run's COLMAP workspace, drops what the `process` stage rebuilds, then runs the pipeline on a
# different mask variant: held-out pass (from `process`), then an eval_mode = all pass (from `train`), with ns-eval
# after each. Both arms of a comparison therefore share camera poses. See README.md in this directory.
set -u

usage() {
cat <<'EOF'
Usage: train_mask_variant_arm.sh --template INI --capture NAME --donor-run ID --arm-run ID --variant NAME [options]

Required
  --template INI       pipeline config to copy settings from (its [dataset] images_subpath, masks and paths are kept)
  --capture NAME       the capture's directory name under {data-root}/colmap and {data-root}/outputs
  --donor-run ID       run id whose COLMAP workspace (sparse/0) is copied
  --arm-run ID         run id of the new arm, PREFIX_YYMMDD-NN-COLMAPVER, e.g. masktest-leaf_261008-01-3120
  --variant NAME       [dataset] mask_variant, i.e. the capture's masks_NAME directory

Options
  --iterations N       max_num_iterations (default 10000)
  --downscale N        downscale_factor, pinned (default 4)
  --train-fraction F   train_split_fraction for the held-out pass (default 0.9)
  --passes LIST        comma list from: frac, all (default frac,all)
  --data-root DIR      default ~/data
  --pipeline-dir DIR   jsps-dt4ag clone (default ~/jsps-dt4ag)
  --dry-run            print what would happen and stop before copying or training
EOF
}

TEMPLATE=""; CAPTURE=""; DONOR=""; ARM=""; VARIANT=""
ITER=10000; DS=4; FRAC=0.9; PASSES="frac,all"; DATA_ROOT="$HOME/data"; PIPE="$HOME/jsps-dt4ag"; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --template) TEMPLATE="$2"; shift 2;;
    --capture) CAPTURE="$2"; shift 2;;
    --donor-run) DONOR="$2"; shift 2;;
    --arm-run) ARM="$2"; shift 2;;
    --variant) VARIANT="$2"; shift 2;;
    --iterations) ITER="$2"; shift 2;;
    --downscale) DS="$2"; shift 2;;
    --train-fraction) FRAC="$2"; shift 2;;
    --passes) PASSES="$2"; shift 2;;
    --data-root) DATA_ROOT="$2"; shift 2;;
    --pipeline-dir) PIPE="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1"; usage; exit 2;;
  esac
done
for v in TEMPLATE CAPTURE DONOR ARM VARIANT; do
  [ -n "${!v}" ] || { echo "missing --$(echo "$v" | tr 'A-Z_' 'a-z-')"; usage; exit 2; }
done
[ -f "$TEMPLATE" ] || { echo "template not found: $TEMPLATE"; exit 2; }
if ! [[ "$ARM" =~ ^(.+)_([0-9]{6})-([0-9]+)-([0-9]+)$ ]]; then
  echo "--arm-run must look like PREFIX_YYMMDD-NN-COLMAPVER (got $ARM)"; exit 2
fi
ID_PREFIX="${BASH_REMATCH[1]}"; RUN_DATE="${BASH_REMATCH[2]}"; RUN_COUNT="${BASH_REMATCH[3]}"; COLMAP_VER="${BASH_REMATCH[4]}"

W="$DATA_ROOT/colmap/$CAPTURE"
OUT="$DATA_ROOT/outputs/$CAPTURE"
EV="$DATA_ROOT/evals/$CAPTURE"
CFG="$DATA_ROOT/configs"
[ -d "$W/$DONOR/sparse" ] || { echo "donor workspace has no sparse model: $W/$DONOR"; exit 1; }
[ -e "$W/$ARM" ] && { echo "REFUSE: arm workspace already exists: $W/$ARM"; exit 1; }
[ -f "$PIPE/pipeline/run_pipeline.py" ] || { echo "no pipeline at $PIPE"; exit 1; }

echo "arm $ARM  variant=$VARIANT  donor=$DONOR  iterations=$ITER  downscale=$DS  passes=$PASSES"
echo "workspace $W/$ARM  outputs $OUT/$ARM  evals $EV"
[ "$DRY" = 1 ] && { echo "(dry run, stopping)"; exit 0; }

export PATH="$PIPE/.venv/bin:$HOME/opt/colmap-prefix/bin:$HOME/.local/bin:$PATH"
mkdir -p "$EV" "$CFG"

mkcfg() { # token eval_mode
  local out="$CFG/$ARM-$1.ini"
  sed -e "s#^mask_variant =.*#mask_variant = $VARIANT#" \
      -e "s#^use_masks =.*#use_masks = true#" \
      -e "s#^id_prefix = .*#id_prefix = $ID_PREFIX#" \
      -e "s#^date =.*#date = $RUN_DATE#" \
      -e "s#^run_count =.*#run_count = $RUN_COUNT#" \
      -e "s#^colmap_version =.*#colmap_version = $COLMAP_VER#" \
      -e "s#^max_num_iterations = .*#max_num_iterations = $ITER#" \
      -e "s#^downscale_factor = .*#downscale_factor = $DS#" \
      -e "s#^eval_mode = .*#eval_mode = $2#" \
      -e "s#^train_split_fraction = .*#train_split_fraction = $FRAC#" \
      "$TEMPLATE" > "$out"
  echo "$out"
}
evalrun() { # token
  local c
  c=$(ls -t "$OUT/$ARM"/splatfacto/*/config.yml | head -1)
  echo "ns-eval $ARM $1 using $c"
  ns-eval --load-config "$c" --output-path "$EV/${ARM}_$1_eval.json" 2>&1 | tail -5
}

cp -a "$W/$DONOR" "$W/$ARM" || { echo "copy failed"; exit 1; }
rm -rf "$W/$ARM/images" "$W/$ARM/images_2" "$W/$ARM/images_4" "$W/$ARM/images_8" "$W/$ARM/transforms.json" "$W/$ARM/sparse_pc.ply"
cd "$PIPE" || exit 1

first=1
for p in $(echo "$PASSES" | tr ',' ' '); do
  case "$p" in
    frac) mode=fraction; tok="evalfrac$(awk "BEGIN{printf \"%d\", (1-$FRAC)*100+0.5}")";;
    all)  mode=all;      tok=evalall;;
    *) echo "unknown pass: $p"; exit 2;;
  esac
  ini=$(mkcfg "$p" "$mode")
  echo; echo "######## $(date +%T) pass $p ($ini)"
  if [ $first = 1 ]; then stage="process"; else stage="train"; fi
  python pipeline/run_pipeline.py --config "$ini" --from-stage "$stage" --run-id "$ARM" || { echo "ABORT: pass $p failed"; exit 1; }
  evalrun "$tok"
  first=0
done
echo; echo "######## $(date +%T) ARM DONE: $ARM"
