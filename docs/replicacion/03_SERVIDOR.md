# 3. Guía paso a paso en el servidor (Slurm + miniconda)

Todo está **sin probar en vuestro cluster** (no tengo acceso): lo que sí se probó está en `DIARIO.md`. Adapta
`slurm/config.sh` a tu cluster (partición, cuenta, tipo de GPU) y ejecuta en este orden.

## 0. Traer el código
```bash
git clone <tu-fork> MT-via-Round-Trip-RL && cd MT-via-Round-Trip-RL
git switch feat/replication
```

## 1. Configurar y crear el entorno (nodo de login)
```bash
nano slurm/config.sh          # PARTITION, ACCOUNT, GRES (gpu:2 o gpu:rtx2080:2), TIME, HF_HOME, CONDA_ENV
bash slurm/setup_env.sh       # crea el entorno conda "mtrl", instala torch 2.8 (CUDA 12.6) + dependencias fijadas
```
* Mira `sinfo -o "%P %G %l"` para saber el nombre de la partición y cómo se llama el recurso GPU.
* Si el driver es antiguo (`nvidia-smi` en un nodo GPU, versión < 525), usa `TORCH_INDEX=https://download.pytorch.org/whl/cu118`.
* Si `setup_env.sh` falla en `check_env.py` porque el login no tiene GPU, es normal: la comprobación real es el paso 3.

## 2. Descargar modelo y datos (nodo con internet)
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate mtrl
export HF_HOME=...            # el mismo de config.sh
python scripts/prefetch.py --models 600M --langs ayr_Latn dyu_Latn wol_Latn rus_Cyrl fur_Latn bho_Deva
ls data/nllb-md               # una carpeta por idioma; run_stage.py las usa solas
```
* El modelo queda en `HF_HOME`; los datos en `data/nllb-md/` (guardados con `save_to_disk`, sin red posterior).
* Si los nodos de cómputo **no tienen internet**, pon `OFFLINE=1` en `slurm/config.sh`.
* **Si `prefetch.py` falla al bajar el dataset** (el script de HF usa enlaces `tinyurl.com`, ver H10), copia desde tu PC
  la carpeta `data/nllb-md/` (ya contiene `ayr_Latn` y `rus_Cyrl`): `rsync -av data/nllb-md/ servidor:ruta/repo/data/nllb-md/`.
  Para los otros idiomas habría que usar la exportación Parquet del dataset en HF (no implementado).

## 3. Comprobar GPUs dentro de un job
```bash
srun --gres=gpu:2 --cpus-per-task=4 --mem=16G --time=00:20:00 [-p PARTICION] \
     bash -c 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate mtrl; nvidia-smi; python scripts/check_env.py'
```
Apunta **modelo y memoria** de las GPUs. Elegir perfil: 8 GB → `--mem tight` (instalar `pip install bitsandbytes`);
11 GB → `--mem lowmem`; ≥24 GB → `--mem paper`. (Estimaciones en 04_CAMBIOS.md; el smoke da el pico real.)

## 4. Lanzar las etapas
```bash
slurm/submit.sh smoke --lang ayr_Latn --mem lowmem       # etapa 1
squeue -u $USER ; tail -f slurm/logs/mtrl-smoke-*.out
python scripts/report.py results/smoke/600M/ayr_Latn/seed27 --no-bootstrap
```
Luego las etapas 2–4 de `02_PLAN.md`. Cada ejecución crea `results/<etapa>/600M/<lang>/seed<N>/` con:
`log.txt` (salida completa), `cmd.txt` (comando exacto), `metrics.jsonl` (chrF++ val/test por paso),
`train_log.jsonl` (una línea por paso: recompensa, KL, forward chrF, s/paso, memoria pico), `hyps/` (traducciones
del test antes/después, para el bootstrap) y `DONE`.

## 5. Límites de las 2 GPUs
Una ejecución usa **exactamente 2 GPUs** (política + referencia). No lances dos a la vez si solo has reservado 2.
Para hacer más con 2 GPUs: perfil `tight` y una ejecución por GPU (`CUDA_VISIBLE_DEVICES=0`, política y referencia
comparten GPU; **no probado**).

## 6. Problemas típicos
| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `CUDA out of memory` en el paso de entrenamiento | perfil de memoria insuficiente | `--mem tight`; o `--set task.training.logp_chunk_groups=1` |
| `CUDA error: no kernel image` | torch sin soporte sm_75 | reinstalar con `cu126`/`cu118` (Turing está soportado) |
| `trust_remote_code` / error al cargar el dataset | `datasets>=4` | usar las versiones de `requirements-server.txt` |
| El job se queda en `Connecting to huggingface.co` | nodo sin internet | `prefetch.py` en login + `OFFLINE=1` |
| `bitsandbytes` falla | falta librería CUDA / versión | usar `--mem lowmem` o `--set task.training.optimizer=adamw` |
