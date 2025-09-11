# ViTCIFAR-Lightning-Example
Repo displaying good pipeline design using lightning to focus on implementation rather than boilerplate

## Run in Docker

Build the image (first time or after dependency changes):

```bash
docker compose build --no-cache
```

Start a training run with the base defaults from `configs/config.yaml`:

```bash
docker compose run --rm train-lightning
```

### Pick configs via CLI

You can select components directly from the CLI without editing files:

```bash
docker compose run --rm train-lightning \
  python -m src.train_lightning \
  model=vit_small data=cifar_10 optim=adamw trainer=lightning
```

### Use curated experiments

Curated combos live under `configs/exp/`. Example:

```bash
docker compose run --rm train-lightning \
  python -m src.train_lightning exp=vit_cifar10
```

This sets the defaults to:

```yaml
defaults:
  - _self_
  - exp: vit_cifar10
```

### Notes

- Hydra output directories default under `./output` or `${OUTPUT_DIR}` if set.
- To see the resolved config at runtime, the script prints it on start.
- You can still override any individual field, e.g. `io.batch_size=256`.
- Note: because `exp` is now selected by default in `configs/config.yaml`, use `exp=quick_debug` to switch experiments. Using `+exp=quick_debug` would add a duplicate and Hydra will error.
