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

## Etapa 3 `pilot` (22-oct, jobs 2410 y 2411): primer resultado de mejora

`ayr_Latn`, NLLB-600M, 600 pasos (1200 frases), `--mem lowmem1` (embeddings congelados, chunk 1), semilla 27,
test completo (1500 frases) antes/después, validación en 200 frases cada 100 pasos. ~36 min por job, 3.4 s/paso.

| Run | Test antes | Test después | Δ | Val (200 frases) por paso | Recompensa de vuelta (primer→último 10 %) | KL (últ. 10 %) |
|---|---|---|---|---|---|---|
| lr 2e-6 (**paper**) | 24.38 | 25.90 | **+1.51** (bootstrap p ≤ 0.001) | 24.49 → 24.56 → 25.06 → 25.33 → 25.50 → 25.60 → 25.71 | 0.1595 → 0.1800 | 0.0001 |
| lr 1e-5 (desviación, ×5) | 24.38 | 27.26 | **+2.87** (p ≤ 0.001) | 24.49 → 25.45 → 25.96 → 26.04 → 26.24 → 26.38 → 26.46 | 0.1661 → 0.2055 | 0.0009 |

Referencia del paper (600M, aimara): 24.38 → 28.13 (**+3.75**) tras ~6000 pasos.

* Pico de memoria con `lowmem1`: **política 7.58 GB, referencia 2.79 GB** (de 11.26 GB): holgura de ~3.7 GB.
* Ambas curvas de validación suben de forma monótona; la recompensa de vuelta sube y el KL se mantiene en ≤ 1e-3.
* Con el lr del paper y solo el 10 % de los pasos ya se recupera ~40 % de la ganancia final del paper (+1.51 de +3.75).
* Intervalos de confianza del bootstrap ≈ ±0.5 chrF++; p = 0.001 es el mínimo posible con 1000 remuestreos.

Cautelas: una sola semilla; sin control RT (MLE) todavía, así que no se separa "efecto RL" de "efecto de entrenar con
round-trip"; el piloto usa embeddings congelados (desviación respecto al paper) y el de lr 1e-5 además un lr 5× mayor.
Decisión: pasar a la etapa 4 (`full`, lr del paper, 2 épocas).

## Decisiones del 22-oct (tarde)

* Pilotos de `dyu_Latn` y `wol_Latn` (lr 2e-6, `lowmem1`): resultados abajo.
* Los runs completos de la noche irán con `--mem lowmem1`. El perfil `exact` (entrenar todo sin congelar embeddings,
  troceo por secuencia) queda aparcado en `02_PLAN.md`, sección "Siguientes pasos", para retomarlo si el run completo
  de aimara queda lejos del paper.
* Una tercera GPU no sirve con el código actual (política en una sola GPU); detalle en `02_PLAN.md` y `04_CAMBIOS.md`.

## Pilotos de `dyu_Latn` y `wol_Latn` (22-oct, lr 2e-6, `lowmem1`, 600 pasos, semilla 27)

Mismas condiciones que el piloto de aimara: test completo (1500 frases) antes/después, validación en 200 frases cada 100 pasos,
3.4 s/paso, pico de memoria 7.58 GiB (política) / 2.79 GiB (referencia), igual que con aimara pese a otro idioma.

| Idioma | Test antes (= Base del paper) | Test después | Δ nuestro | Δ paper (final, ~6000 pasos) | Δ nuestro / Δ paper | Val (200 frases) por paso | Recompensa de vuelta (1.º→últ. 10 %) |
|---|---|---|---|---|---|---|---|
| `ayr_Latn` | 24.38 (24.38) | 25.90 | +1.51 | +3.75 | 40 % | 24.49 → 25.71 | 0.1595 → 0.1800 |
| `dyu_Latn` | 18.03 (18.03) | 19.90 | **+1.87** | +4.47 | 42 % | 17.10 → 18.37 | 0.1557 → 0.1780 |
| `wol_Latn` | 20.74 (20.73) | 21.91 | +1.17 | +3.23 | 36 % | 20.16 → 20.76 | 0.1634 → 0.1828 |

* Los tres mejoran con bootstrap pareado significativo (p ≤ 0.001, intervalos ≈ ±0.4–0.5) y con la recompensa de vuelta y la
  chrF de ida al alza; el KL se queda en ≤ 2e-4.
* Con solo el 10 % de los pasos los tres recuperan una fracción parecida (36–42 %) de la ganancia final del paper. Es una
  coincidencia llamativa pero no prueba que el resto de la curva sea lineal: puede saturar.
* En `wol_Latn` la validación (200 frases) se aplana desde el paso 200 (20.69 → 20.76), aunque el test sube +1.17: puede ser
  ruido del subconjunto de validación, pero es el idioma con menos señal; en `dyu_Latn` la validación sigue subiendo despacio.
* Elección para los runs completos de la noche: `ayr_Latn` y `dyu_Latn` (mayor Δ del paper y mayor Δ del piloto, respectivamente).

## Etapa 4 `full` (22→23-oct, noche): réplica de `ayr_Latn` y `dyu_Latn` con el paper

Config del paper (lr 2e-6, 2 épocas = 6000 pasos, batch 2, K=4, β=0.04, ε=0.2, referencia cada 16 pasos, T=1.8, top-k 100,
top-p 0.95), NLLB-600M, semilla 27, `--mem lowmem1` (única desviación: embeddings congelados), test completo de 1500 frases
antes/después, validación en 300 frases cada 500 pasos. 3.4 s/paso, pico de memoria 7.59 GiB (política) / 2.79 GiB (referencia).

| Idioma | Base nuestro (paper) | **Ours nuestro** | Ours paper | Δ nuestro | Δ paper | Bootstrap pareado antes/después |
|---|---|---|---|---|---|---|
| `ayr_Latn` | 24.38 (24.38) | **28.08** | 28.13 | **+3.69** | +3.75 | 24.4 ± 0.5 → 28.1 ± 0.4, p ≤ 0.001 |
| `dyu_Latn` | 18.03 (18.03) | **22.62** | 22.50 | **+4.58** | +4.47 | 18.0 ± 0.4 → 22.6 ± 0.4, p ≤ 0.001 |

Las diferencias frente a la tabla del paper son −0.05 (`ayr`) y +0.12 (`dyu`), muy por debajo de los intervalos de
confianza del bootstrap (±0.4–0.5): **los resultados son compatibles con los publicados** (no demuestra igualdad exacta).

Curvas de validación (300 frases, chrF++), cada 500 pasos:
* `ayr`: 24.91, 25.77, 26.24, 26.57, 26.49, 26.81, **26.92 (paso 3000 = fin de la 1.ª época)**, 26.97, 27.14, 27.28, 27.36, 27.59, 27.60.
* `dyu`: 18.08, 19.66, 20.44, 20.78, 21.22, 21.54, **21.72 (paso 3000)**, 21.85, 21.98, 22.13, 22.33, 22.40, 22.80.

Observaciones:
* Mejora casi monótona en ambos; el cambio de época (paso 3000), que era el riesgo sin probar en GPU, no rompió nada
  (checkpoint guardado, entrenamiento continuo). No hay colapso ni *reward hacking* visible: la recompensa de vuelta pasa de
  0.170 → 0.246 (`ayr`) y 0.168 → 0.229 (`dyu`), el KL se queda en ~1e-4 y la chrF de ida contra la referencia real
  sube (18.7 → 22.7 y 15.3 → 18.6).
* Los pilotos de 600 pasos daban ~40 % de la ganancia final (+1.51 y +1.87): la curva no era lineal pero tampoco saturaba pronto.
  `dyu` sigue subiendo en el último tramo; `ayr` se aplana al final (27.59 → 27.60).
* Cautelas: una sola semilla (igual que el paper); no hay control RT, así que no se separa el efecto del RL del de entrenar con
  round-trip; la desviación de congelar embeddings no se ha aislado, pero no ha impedido reproducir el resultado. Por eso el
  perfil `exact` aparcado en `02_PLAN.md` **no se activa**: su condición (quedar lejos del paper) no se ha dado.

