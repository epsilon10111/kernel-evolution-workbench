set -eu
: "${SSH_CONNECTION:?Run this gate over SSH}"
umask 077

gpu_uuid=$1
image=$2
problem=$3
[[ "$gpu_uuid" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] || exit 64
[[ "$image" =~ ^[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$ ]] || exit 64
[[ "$problem" =~ ^[a-z][a-z0-9_]*$ ]] || exit 64
case "$4" in
    copy) driver=(local_eval.py copy --problem "$problem") ;;
    batch)
        [[ "$5" =~ ^[1-4]$ && "$6" =~ ^[0-3](,[0-3]){0,3}$ ]] || exit 64
        driver=(local_eval.py batch --problem "$problem" --order "$6")
        for ((i=0; i<$5; ++i)); do driver+=(--source "candidate-$i.cu"); done ;;
    *) exit 64 ;;
esac

uid=$(id -u)
test "$uid" != 0 || { echo 'Use a non-root SSH account for GPU evaluation' >&2; exit 64; }
inventory=$(timeout -k 2 30 nvidia-smi --id="$gpu_uuid" --query-gpu=uuid,name,pci.bus_id,driver_version --format=csv,noheader)
IFS=, read -r observed gpu pci driver_version <<< "$inventory"
gpu=${gpu# }; pci=${pci# }; driver_version=${driver_version# }
test "$observed" = "$gpu_uuid" && test "$gpu" = 'NVIDIA H20' || {
    echo 'The configured GPU UUID did not resolve to an H20' >&2; exit 64; }
# This lock is shared by all workbench users of the same physical GPU.
exec 9<"/sys/bus/pci/devices/${pci: -12}"
flock -n 9 || { echo 'The GPU is locked by another evaluation' >&2; exit 75; }
processes=$(timeout -k 2 30 nvidia-smi --id="$gpu_uuid" --query-compute-apps=gpu_uuid --format=csv,noheader)
if grep -Fxq "$gpu_uuid" <<< "$processes"; then echo 'The GPU already has compute work' >&2; exit 75; fi
image_id=$(docker image inspect --format '{{.Id}}' "$image")

job=$(mktemp -d /tmp/kernel-workbench.XXXXXXXX)
trap 'rm -rf -- "$job"' EXIT
cd "$job"
tar -xf -
mkdir scratch
python3 - "$gpu_uuid" "$gpu" "$driver_version" "$image" "$image_id" <<'PY'
import json
import sys
from pathlib import Path

uuid, gpu, driver, digest, image_id = sys.argv[1:]
Path("host.json").write_text(json.dumps(dict(gpu_uuid=uuid, gpu=gpu, driver_version=driver,
    image_digest=digest, image_id=image_id, remote_ssh=True)) + "\n")
PY
chmod 444 host.json ./*.py
runtime=(docker run --rm --pull=never --network none --cap-drop ALL --security-opt no-new-privileges=true
    --user "$uid:$(id -g)" --read-only --gpus "device=$gpu_uuid"
    --tmpfs "/tmp:rw,nosuid,nodev,size=128m" --mount "type=bind,src=$job,dst=$job,readonly"
    --mount "type=bind,src=$job/scratch,dst=$job/scratch"
    --mount "type=bind,src=$job/host.json,dst=/run/kernel-workbench-gpu.json,readonly"
    --workdir "$job" -e "HOME=$job/scratch" -e "TMPDIR=$job/scratch"
    -e "CUDA_VISIBLE_DEVICES=$gpu_uuid" -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e "XDG_CACHE_HOME=$job/scratch"
    -e "CUDA_CACHE_PATH=$job/scratch/cuda"
    --entrypoint /usr/bin/env "$image")
result=0
"${runtime[@]}" timeout -k 10s 900s python "${driver[@]}" --output "$job/scratch/report.json" 9>&- 1>&2 || result=$?
if test -f "$job/scratch/report.json"; then
    cat "$job/scratch/report.json"
fi
exit "$result"
