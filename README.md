# Kernel Evolution Workbench

Develop CUDA kernels against a fixed correctness reference and compare candidates on one GPU. The workbench queues paired measurements, records the exact source and raw samples for each run, and provides a local dashboard for inspecting results.

## Try it without a GPU

Use Linux and Python 3.11 or newer. From the repository root, inspect the example kernel's ABI and run the core regression tests.

~~~sh
python3 -m harness.local_eval manifest --problem leaky_relu
python3 -m unittest harness.test_local_campaign harness.test_local_loop harness.test_local_runner
~~~

The full test command is `python3 -m unittest discover -s harness -p 'test_*.py'`. Some tests use PyTorch, a C++ compiler, Playwright with Chromium, or Bubblewrap.

## Run a benchmark

On a machine with one visible NVIDIA H20, PyTorch with CUDA support, `nvcc` and `nvidia-smi`, benchmark the included Leaky ReLU baseline.

~~~sh
mkdir -p runs
python3 -m harness.local_eval run --problem leaky_relu \
  --source examples/leaky_relu.cu --output runs/baseline.json
python3 -m harness.local_eval recompute --report runs/baseline.json
~~~

The standalone evaluator loads the compiled kernel into its Python process. Run it with source you trust. To add a workload, define its CUDA ABI, input generator, reference, and shapes in [local_eval.py](harness/local_eval.py). The other built-in manifests are `hinge` and `histogram`.

## Run an iteration campaign

The queue runner uses a Linux H20 host reachable over SSH. The SSH account needs Docker with NVIDIA GPU support, Python 3, and a locally installed container image pinned by digest. The image must contain CUDA-enabled PyTorch, `nvcc`, and the coreutils `timeout` command. Configure host keys through normal SSH setup. The workbench locks the selected physical GPU for each measurement group and runs the candidates in a container without networking.

~~~sh
export KEW_SSH_HOST='<ssh-host-alias>'
export KEW_GPU_UUID='<gpu-uuid>'  # See nvidia-smi -L on the runner.
export KEW_CUDA_IMAGE='<image-reference>@sha256:<digest>'
# Optional. OpenSSH uses its default configuration when KEW_SSH_CONFIG is unset.

campaign=runs/leaky-relu/campaign
python3 -m harness.local_campaign --root "$campaign" init leaky_relu \
  examples/leaky_relu.cu --contract 'FP32 Leaky ReLU with the manifest ABI'
python3 -m harness.local_runner drain --root "$campaign" --watch
~~~

From a second terminal, open the local dashboard or start the optional model workers.

~~~sh
python3 -m harness.dashboard --campaign runs/leaky-relu/campaign
python3 -m harness.local_loop --root runs/leaky-relu/campaign --rounds 3
~~~

The dashboard binds to `http://127.0.0.1:8765/`. Run the dashboard and the model workers in separate terminals. The workers require Bubblewrap and an authenticated Codex CLI with `gpt-6.2-astra` and the `max` reasoning tier available.

For manual iteration, use `begin a CARD.json` with an experiment card, `submit a CANDIDATE.cu` with a complete CUDA file, and `review a CONCLUSION` with a conclusion after evaluation. The commands are available through `python3 -m harness.local_campaign --help`. The `context a --compact` command shows candidate lineage, local reports, and shared findings. The `attach-profile SOURCE_SHA SHAPE_ID REPORT.md` command associates a Markdown profiler report with an archived source and a shape.

## How a candidate is selected

Each measurement leg checks the entire frozen input set, records CUDA Event and end-to-end times separately, and aggregates the median GPU time per input, the mean per shape, and the sum across shapes. Three baseline self-comparisons establish a noise band. A candidate is then measured against the current incumbent in two groups, in `ABBA` and `BAAB` order. Both gains must clear the noise band for whole-workload promotion.

The experiment handoff and profiling workflow draws on the [FP8 MoE experiment protocol](https://github.com/DongyunZou/fp8-moe-example-workspace/blob/60fc0c94fa2a550b0f9d524d8748e6662ae9c5d6/docs/EXPERIMENT_PROTOCOL.md).
