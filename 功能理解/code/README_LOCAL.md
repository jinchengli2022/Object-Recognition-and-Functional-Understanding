# Local GEAL Reproduction Notes

This directory is a local reproduction of the official GEAL repository:
`https://github.com/DylanOrange/geal`.

## What is already prepared

- Official source code is cloned under `3d_affordance/GEAL`.
- The missing root `.gitmodules` entry for `thirdparty/diff-gaussian-rasterization/third_party/glm`
  is added locally so `git status` and submodule checkout work.
- Official checkpoint download helper:
  `tools/download_ckpts.sh`.
- Dataset-style evaluation configs:
  `config/evaluation_piad_seen.yaml`,
  `config/evaluation_piad_unseen.yaml`,
  `config/evaluation_laso_seen.yaml`,
  `config/evaluation_laso_unseen.yaml`.
- Hand-written object-label to salient-affordance table:
  `config/object_affordance_map.tsv`.
- Custom single-point-cloud inference wrapper:
  `tools/geal_infer_pointcloud.py`.

## Environment

The official README says GEAL was tested with Python 3.10, CUDA 11.8, and
PyTorch 2.1.0. For local inference, the existing conda environment
`sam3d-objects` already has the key runtime packages:

```bash
source /home/amadeus/anaconda3/etc/profile.d/conda.sh
conda activate sam3d-objects
```

I also installed `socksio` in `sam3d-objects`, because the local Hugging Face
client uses a SOCKS proxy and otherwise fails before downloading `roberta-base`.

For full stage-1/stage-2 training, follow the official environment exactly and
build `thirdparty/diff-gaussian-rasterization`. The custom point-cloud inference
wrapper only uses Branch3D, so it does not require the rasterizer extension.

## Download checkpoints

```bash
cd /home/amadeus/project_dev/3d_affordance/GEAL
./tools/download_ckpts.sh
```

Expected files:

```text
ckpt/piad_seen.pt
ckpt/piad_unseen.pt
ckpt/laso_seen.pt
ckpt/laso_unseen.pt
```

## Official dataset evaluation

After preparing LASO or PIAD data in the format described by the official
README, run:

```bash
cd /home/amadeus/project_dev/3d_affordance/GEAL
python scripts/evaluation.py --config config/evaluation_laso_seen.yaml --device cuda
python scripts/evaluation.py --config config/evaluation_piad_seen.yaml --device cuda
```

## Custom PLY inference

This wrapper runs every affordance prompt on the same object point cloud and
saves all scores, not only argmax labels. By default it saves the sampled model
input points only; use `--output-points original` only if you explicitly want
KNN interpolation back to the original dense cloud.

Example:

```bash
cd /home/amadeus/project_dev/3d_affordance/GEAL
python tools/geal_infer_pointcloud.py \
  --input /home/amadeus/project_dev/object_model/model_results/demo_object_data/kettle/kettle_geom.ply \
  --ckpt ckpt/laso_seen.pt \
  --config config/evaluation_laso_seen.yaml \
  --out-dir /home/amadeus/project_dev/object_model/aff_results/geal_kettle \
  --name kettle_geal \
  --object-class kettle \
  --affordances auto \
  --num-points 2048 \
  --sampling random \
  --batch-size 4 \
  --output-points sampled \
  --min-affordance-p95 0.35 \
  --clean-components \
  --min-component-points 40 \
  --min-component-ratio 0.05 \
  --keep-components 1 \
  --save-argmax \
  --save-heatmaps
```

Outputs:

- `*_geal_scores.npz`: `points` and `scores` with shape `[num_output_points, num_kept_affordances]`.
- `*_geal_meta.json`: affordance names, checkpoint, normalization metadata.
- `*_geal_argmax.ply`: one color per highest-scoring affordance.
- `*_heatmaps/*.ply`: one red-gray heatmap per affordance.

`--min-affordance-p95` drops affordance classes whose 95th-percentile score is
too low, so weak global classes do not appear in the color map or PLY outputs.
`--clean-components` removes small disconnected argmax regions. With
`--keep-components 1`, each affordance keeps only its largest connected region;
removed points are colored gray and saved as `argmax_labels=-1` in the NPZ.

Important: GEAL is language-conditioned internally. The wrapper fixes the
affordance vocabulary and prompts, so the user-facing input can still be just
one point cloud.
