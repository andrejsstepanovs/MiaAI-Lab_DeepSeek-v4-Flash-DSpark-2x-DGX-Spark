# AGENTS.md — DeepSeek V4 Flash DSpark cluster (2x DGX Spark)

Practical guidance for working on this repo and the running cluster. Read the
README and `docs/` for full context; this file captures the operational
lessons from a full upgrade cycle (see git history around Aug 2026).

## Topology & remotes

- **2-node cluster.** Head node runs `start-deepseek-v4-flash-dspark.sh` and
  SSHes to the worker `spark2` (see `WORKER_HOST` in `.env.dspark`). Both nodes
  have their own copy of this repo at `~/ds4-recipe`.
- The head's `start-*.sh` **auto-syncs** to the worker at every start:
  `docker-compose.dspark.yml`, `.env.dspark`, `docker-compose.vl-sidecar.yml`,
  `recipe/.../dspark_proposer.py`, and the whole `patches/` dir. You do not
  normally need to copy anything to spark2 manually.
- Remotes: `origin` = upstream `MiaAI-Lab/...`, `andrejs` = personal fork.
  Local `main` tracks `origin/main`.

## THE ONE GOLDEN RULE

**`DSPARK_VLLM_IMAGE` in `.env.dspark` stays pinned to the prebuilt Anemll
image.** Do not switch it to the local Stage-C build for an "upgrade".

```
DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8
```

Why: the entire hotfix suite (`patches/`) and the compose entrypoint are written
and tested **only against this image**. The local Stage-C image is a different
vLLM build that breaks them:

| Issue on Stage-C image | Why |
|---|---|
| `cp: cannot create ... deepseek_v4_encoding.py: No such file or directory` | vLLM lives at `/opt/env/lib/python3.12/site-packages`, not `/usr/local/lib/python3.12/dist-packages` |
| `bash: line N: python3: command not found` | compose runs `bash -lc`; the image's login profile clobbers PATH, dropping `/opt/env/bin` |
| `[FAIL] streaming final chunk: anchor not found in serving.py` (issue55) | Stage-C vLLM source differs from what the hotfix anchors match |

The Stage-C lane is experimental. Only touch it if you explicitly intend to
develop/benchmark it, never as part of a routine upstream upgrade.

## Upgrading to latest upstream (the safe way)

1. Inspect first: `git fetch origin`, then `git log --oneline HEAD..origin/main`.
2. Merge upstream into local `main`:
   `git merge origin/main` (keeps your local commits; this repo keeps local
   setup commits on `main`).
3. Mirror to the fork: `git push andrejs main` and push any new upstream
   branches to `andrejs` (e.g. `git push andrejs origin/feat/x:refs/heads/feat/x`).
4. **Do not rebuild any image** for the mainline hotfixes. Upstream hotfixes are
   applied at container start from the mounted `patches/` + the compose
   entrypoint — the prebuilt Anemll image already has them on disk.
5. Restart the stack so the merged files actually take effect:
   - `./stop-deepseek-v4-flash-dspark.sh` (required if containers already
     exist; `start-*.sh` exits 3 "already up" otherwise)
   - `./start-deepseek-v4-flash-dspark.sh`
   The start script re-syncs compose/`.env`/patches to spark2 automatically.
6. Verify: `./status-deepseek-v4-flash-dspark.sh`, then confirm the running
   image is still the Anemll digest pin on **both** nodes
   (`docker ps` + check worker via ssh).

## Checking whether the pinned Anemll image is still current

The digest pin is immutable — you stay on the tested build until you bump it.
To see if Anemll has published anything newer:

- List tags: `ghcr.io/anemll/dspark-vllm-gx10` currently has only `0.1.0` and
  `0.1.1` (`0.1.1` is latest). Get the manifest digest of a tag:
  ```
  TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:anemll/dspark-vllm-gx10:pull" | jq -r .token)
  curl -sI -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    "https://ghcr.io/v2/anemll/dspark-vllm-gx10/manifests/0.1.1" | grep -i docker-content-digest
  ```
- If the digest equals the `.env.dspark` pin (`sha256:a83948...`), you're on the
  current build — no pull needed. If a new tag or a re-pointed `0.1.1` digest
  appears, you *may* bump the pin — but the hotfix suite is only guaranteed
  against the tested digest; validate on a bench before adopting.

## What NOT to do

- **Do not** change `DSPARK_VLLM_IMAGE` to activate a single upstream fix
  (e.g. PR #127 "shared expert loading") — that fix is Stage-C-only and is
  irrelevant on the Anemll image. You were already running fine.
- **Do not** manually copy files to spark2. `start-*.sh` syncs what is needed.
  The worker's git checkout is intentionally stale and unused at runtime.
- **Do not** run `./build-dspark-vllm-runtime.sh` unless you really want the
  Stage-C image. If you do, know that it sources `.env.dspark` and reuses
  `DSPARK_VLLM_IMAGE` as the final `-t` tag, so with the digest pin it fails:
  `ERROR: refusing to create a tag with a digest reference`. Per
  `docs/ENVS.md` you must first set
  `DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c` — which also
  changes what the cluster will run next start. Put it back afterward.
- **Do not** assume the Stage-C image is a drop-in replacement (see table above).
- **Do not** edit the untracked local files (`dspark-autostart.sh`,
  `dspark.service`, `session-*.md`) or commit them unless asked.

## Housekeeping

- The local Stage-C build chain (`vllm-dspark-runtime:*`) is ~91 GB per node
  when present. It can be deleted on both nodes to reclaim space if unused:
  `docker image prune` (or remove the specific tags). The Anemll image is
  pulled from ghcr and must stay.
- Reverting to a known-good state = restore `.env.dspark`,
  `docker-compose.dspark.yml`, and any image Dockerfile changes from git:
  `git checkout -- .env.dspark docker-compose.dspark.yml` (adjust paths).
