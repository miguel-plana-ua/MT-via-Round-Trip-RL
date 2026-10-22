# Diario de trabajo (22-oct-2026)

Registro de lo hecho, en orden, y de **qué se verificó y qué no**.

## Pasos

1. **Lectura del repo** (rama `src-tgt-src-outcome`): `main.py`, `utils.py`, `dl.py`, `configs/`, `baselines/`.
   También leí, solo para entender qué se intentó, la documentación y el código de `feat/try` y comparé ramas
   `upstream/*`. La rama `grpo_..._aym` de upstream es experimental (K=6, acumulación 16, recompensa con Goldfish): **no**
   es la configuración del paper; la rama por defecto sí coincide con los hiperparámetros del paper.
2. **Paper** (arxiv.org/html/2601.12535): hiperparámetros, Tabla 1, ablación de recompensas, fluidez. Ver `01_ANALISIS.md`.
3. **Rama de trabajo.** Primero creé una rama desde `feat/try` por error; la descartaste (la quieres desde la original).
   Borré esa rama (sin commits) y creé `feat/replication` desde `src-tgt-src-outcome`. `feat/try` queda intacta; no
   se ha usado ninguno de sus cambios (los dos cambios equivalentes, p. ej. el *shim* de `dl.py`, están reescritos).
4. **Código** (detalle en `04_CAMBIOS.md`): `utils.py` (log-prob con logsumexp + modo de `backward` por trozos),
   `main.py` (logs en disco, hipótesis, opciones de memoria, refresco de la referencia in-place, `max_steps`),
   `dl.py` (shims, subconjuntos aleatorios, copia local del dataset), `configs/task/nllb_md.yaml` (modelo público +
   opciones nuevas con defaults del original).
5. **Scripts**: `scripts/run_stage.py`, `report.py`, `check_env.py`, `prefetch.py`; `slurm/*`; `tests/test_grpo_loss.py`.
6. **Documentación** en `docs/replicacion/`.

## Verificado (en mi máquina, Windows, CPU, torch 2.8.0+cpu)

| Qué | Cómo | Resultado |
|---|---|---|
| El nuevo `utils.py` calcula lo mismo que el original | Comparación numérica contra `utils.py` del commit `3d9935d` con un M2M100 diminuto aleatorio | Pérdida, métricas y gradientes idénticos (tol. 1e-6) |
| El modo por trozos (`logp_chunk_groups` 1/2/3, `backward_scale` 0.5) | `tests/test_grpo_loss.py` | Idéntico al modo normal (pérdida, logs, gradientes) |
| `ref.load_state_dict(model.state_dict())` deja KL=0 | mismo test | OK (con embeddings atados) |
| `main.py` extremo a extremo con el modelo real 600M en CPU (aimara, 2 frases, 1 paso, `logp_chunk_groups=1`, `freeze_embeddings=true`, `batch_size=1`, K=2) | ejecución real | Termina con exit 0; escribe `metrics.jsonl`, `train_log.jsonl`, `hyps/`. Parámetros entrenables con embeddings congelados: **352,718,848 de 615,073,792 (57.35 %)**. En el paso 1: KL = 0 y pérdida ≈ −6e-7 (esperado: ratio 1 y ventajas de media 0). Un paso tardó 343 s **en CPU** (irrelevante para GPU). |
| Modo `eval.only` (etapa `base`) | ejecución real offline con la copia local del dataset (4 frases val/test) | Termina; evalúa el test **una** sola vez; ya no repite |
| `scripts/run_stage.py` (dry-run y ejecución), `scripts/report.py` (tabla y bootstrap) | sobre las salidas anteriores | Funcionan; el bootstrap pareado se omite si antes/después son idénticos (sacrebleu da un p "significativo" espurio en ese caso) |
| Scripts de Slurm | `bash -n` + finales de línea LF | Sintaxis OK |
| Datos | inspección | train 6000 / valid **1309** / test **1500**; ordenado por dominio chat→news→health |

## NO verificado (hay que hacerlo en el servidor)

* **Nada de lo que depende de la GPU**: memoria pico, s/paso, perfil `lowmem`/`tight`, `bitsandbytes`/AdamW 8-bit,
  reparto política `cuda:1` / referencia `cuda:0` y la copia entre GPUs de `load_state_dict`.
* El refresco de la referencia dentro del bucle real (paso 16): solo probado en test unitario; el smoke de servidor
  (20 pasos) lo cruza.
* `slurm/*.sh` en un Slurm real (partición, GRES, módulos), y `setup_env.sh` (wheels `cu126` para Turing/sm_75).
* Los valores de la Tabla 1 del paper (extraídos automáticamente del HTML): contrastar con el PDF.
* Que `prefetch.py` pueda bajar `dyu_Latn`, `wol_Latn`, `fur_Latn`, `bho_Deva`: el script del dataset usa enlaces
  `tinyurl.com`; desde mi red dejaron de responder (la red local además intercepta TLS: `self-signed certificate`), así que
  no pude descargar esos 4 idiomas. Las copias locales de `ayr_Latn` y `rus_Cyrl` en `data/nllb-md/` se generaron a partir
  de la caché de HF que ya existía en este PC.
* Ningún experimento de entrenamiento real ha corrido: **no hay todavía ningún resultado de replicación**.

## Estado del repositorio
Rama `feat/replication`, cambios **sin commitear** (se ven con `git status` / `git diff`). `data/` y `results/` están
en `.gitignore`.

## Primera ejecución en el servidor `abaco` (22-oct, job 2403, smoke)

Entorno: 1 nodo, 8 × RTX 2080 Ti (11 GB), driver 575, partición `compute` sin límite, 20 cores / 125 GB compartidos.
Slurm fija `CUDA_VISIBLE_DEVICES=0,1` pero **no aísla las GPUs** (sin cgroup de dispositivos): torch ve solo 2, `nvidia-smi`
ve 8. Por eso `slurm/submit.sh` serializa los jobs (`--dependency=singleton`) y se lanza siempre vía Slurm.

Resultado del smoke (`--mem lowmem`, 20 pasos, copia local del dataset `data/nllb-md`):

| Medida | Valor |
|---|---|
| Duración total del job | 1 min 29 s, exit 0, sin errores en `log.txt` ni `.err` |
| s/paso (mediana) | **3.3** → piloto 600 pasos ≈ 33 min; completo 6000 pasos ≈ 5.5 h (+ evaluaciones) |
| Pico de memoria (tensores PyTorch) | **referencia (cuda:0) 3.27 GB; política (cuda:1) 8.9 GB de 11.26 GB** |
| KL a la referencia | 0 hasta el paso 16 (refresco) y luego 3e-6 → 7.6e-6 → 2e-5: los pesos sí se actualizan |
| chrF++ test 32 frases antes/después | 27.83 / 27.80 (ruido; no se espera mejora en 20 pasos) |

Decisión: margen de memoria justo (~1.5 GB tras contexto CUDA y fragmentación) → perfil `lowmem1` (chunk 1) y
`expandable_segments` en el job para los runs largos. Aún no se ha comprobado que el pico no crezca en runs largos.

## Etapa 2 `base` (22-oct, jobs 2404–2409): el Base reproduce la Tabla 1 del paper

Evaluación sin entrenar, test completo (1500 frases), greedy, `max_new_tokens=63`, NLLB-600M, 6 jobs serializados:

| Idioma | Base (nuestro) | Base (paper) | Diferencia |
|---|---|---|---|
| `ayr_Latn` | 24.38 | 24.38 | 0.00 |
| `dyu_Latn` | 18.03 | 18.03 | 0.00 |
| `fur_Latn` | 45.75 | 45.75 | 0.00 |
| `wol_Latn` | 20.74 | 20.73 | +0.01 |
| `bho_Deva` | 43.08 | 43.07 | +0.01 |
| `rus_Cyrl` | 49.22 | 48.80 | +0.42 |

Conclusión: datos (test de 1500 frases), tokenización, decodificación (greedy) y métrica (chrF++) coinciden con los del
paper; el "1600" del paper era aproximado. Por tanto, **los deltas Base→Ours medidos aquí son comparables con los del
paper**. Solo `rus_Cyrl` se desvía algo (+0.42), sin explicación; ruso es el idioma donde el paper ya difiere del resto.
Tiempo: los 6 jobs tardaron claramente menos de lo estimado (no hay `sacct`; no se midió con exactitud).
Nota: los logs de Slurm se llaman `slurm/logs/mtrl-<etapa>-<jobid>.{out,err}`.
