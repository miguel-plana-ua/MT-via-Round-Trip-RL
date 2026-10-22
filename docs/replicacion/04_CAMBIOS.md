# 4. Cambios respecto al repositorio original

Rama `feat/replication`, creada desde `src-tgt-src-outcome` (`3d9935d`). Ver el diff exacto con
`git diff src-tgt-src-outcome..feat/replication`.

**Regla:** con la configuración por defecto el entrenamiento hace lo mismo que el código original. Todo lo que cambia
el cómputo es **opt-in** y se activa solo con los perfiles `--mem lowmem|tight` de `scripts/run_stage.py`.

## 4.1 Cambios obligatorios (sin ellos no corre / no es la réplica)

| Fichero | Cambio | Por qué |
|---|---|---|
| `configs/task/nllb_md.yaml` | `model.name` = `facebook/nllb-200-distilled-600M` (el original apuntaba al checkpoint local del autor, comentado) | Partir del modelo público (hallazgo H1) |
| `dl.py` | `torchdata`/`pytorch_lightning` se importan con `try/except`; si no están, se usan `torch.utils.data.Dataset` y un sustituto mínimo de `save_hyperparameters` | Esos paquetes ya no son compatibles con torch moderno; comportamiento idéntico (H9) |

## 4.2 Cambios neutrales (no alteran el entrenamiento)

| Fichero | Cambio | Por qué |
|---|---|---|
| `main.py` | Escribe `metrics.jsonl` (chrF++ de val/test con paso y nº de frases), `train_log.jsonl` (una línea por paso) y `hyps/*.txt` (traducciones del test antes/después) en el directorio de salida de Hydra | Resultados sin depender de wandb; bootstrap pareado |
| `main.py` | `step_idx` inicializado; en `eval.only` no se repite la evaluación del test; las evaluaciones usan el paso global (el original reiniciaba `step_idx` cada época) | Bugs latentes (H7) |
| `main.py` | Checkpoint por época dentro del directorio Hydra (original: `./model` en el cwd) y desactivable con `training.save_model=false` | No llenar disco / no pisar ejecuciones |
| `main.py` | Refresco del modelo de referencia con `ref_model.load_state_dict(model.state_dict())` en lugar de `deepcopy(model).to(aux)` | Mismo resultado; el `deepcopy` creaba momentáneamente un 2º modelo completo en la GPU de la política |
| `utils.py` | Log-prob por token como `logit[target] - logsumexp(logits)` en lugar de `log_softmax` + `gather` | Mismo valor (test unitario); evita materializar un 2º tensor (B,L,256k) |
| `dl.py` | Si `data.path` es un directorio con `<config>/` guardado por `prefetch.py`, se carga con `load_from_disk` (sin tinyurl); si no, se usa `load_dataset` como antes | El loader original necesita `tinyurl.com` en cada ejecución, incluso offline (H10) |
| `dl.py` | `max_train/val/test_samples` + `subset_seed`: subconjuntos **aleatorios con semilla** | Pruebas pequeñas reproducibles; evita el sesgo de coger las primeras N filas (H5) |

## 4.3 Opcionales que SÍ cambian el cómputo (solo con `--mem lowmem|tight`)

| Opción (`task.training.*`) | Efecto | Desviación respecto al paper |
|---|---|---|
| `logp_chunk_groups=N` | La pérdida GRPO y su `backward` se calculan N prompts cada vez (recompensas y ventajas se calculan antes, por grupo, igual que siempre) | **Ninguna** matemáticamente: `tests/test_grpo_loss.py` comprueba que pérdida, métricas y gradientes coinciden con el modo normal |
| `freeze_embeddings=true` | Congela el embedding compartido y `lm_head` (262M de 615M parámetros) | **Sí**: esas matrices no se actualizan. Con lr 2e-6 se mueven muy poco, pero es una desviación real |
| `optimizer=adamw8bit` | AdamW de 8 bits (`bitsandbytes`) | **Sí**: el estado del optimizador se cuantiza |
| `max_steps` | Para tras N batches | Solo para pilotos |

Perfiles de `run_stage.py`: `paper` = nada de lo anterior; `lowmem` = `freeze_embeddings` + chunk de 2 prompts;
`tight` = `freeze_embeddings` + chunk de 1 + `adamw8bit`.

Estimación (mía, a verificar con el pico de memoria del smoke) de la GPU de la política: `paper` ≈ 14–16 GB,
`lowmem` ≈ 8–9 GB, `tight` ≈ 5–6 GB; la GPU de la referencia necesita ≈ 2.5 GB + logits.

## 4.4 Ficheros nuevos

| Fichero | Para qué |
|---|---|
| `scripts/run_stage.py` | Lanza `main.py` con los presets de cada etapa (smoke/base/pilot/full) y perfil de memoria |
| `scripts/report.py` | Resumen de una ejecución: chrF++ antes/después frente al paper, bootstrap pareado, curvas |
| `scripts/check_env.py`, `scripts/prefetch.py` | Comprobar entorno/GPUs; descargar modelo y guardar los datos en `data/nllb-md/` |
| `slurm/` | `config.sh`, `setup_env.sh`, `submit.sh`, `job.sbatch`, `requirements-server.txt` |
| `tests/test_grpo_loss.py` | Test en CPU con un M2M100 diminuto: equivalencia del modo por trozos |
| `docs/replicacion/` | Esta documentación |
| `.gitattributes` | Finales de línea LF en `.sh/.sbatch/.py` (el repo se edita en Windows; CRLF rompe los scripts en Linux) |

## 4.5 Lo que NO se ha tocado
Hiperparámetros de `training` (lr 2e-6, batch 2, K=4, β, ε, T, top-k/p, refresco de ref cada 16), la recompensa (solo
chrF++), el bucle GRPO, la generación, la evaluación (greedy, 63 tokens) y los **baselines** (`baselines/`), que siguen
usando rutas del autor (`/workspace/miniconda3/...`) y entrenan con AdamW fp32 completo (no caben en 8–11 GB). Se
abordarán en la etapa 5.
