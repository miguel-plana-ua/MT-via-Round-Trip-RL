# 1. Análisis del repositorio y diagnóstico

Base de trabajo: rama `src-tgt-src-outcome` (commit `3d9935d`, la rama por defecto del repo del autor). **No** se ha
usado nada de `feat/try`; la rama de trabajo es `feat/replication`, creada desde la original.
Paper: <https://arxiv.org/abs/2601.12535> (Attia & Aji, 2026).

## 1.1 Qué hace el código (verificado leyéndolo)

Un único modelo NLLB-200-distilled (600M o 1.3B) se afina **sin usar la mitad objetivo del corpus paralelo**:

1. `main.py`: para cada batch de 2 frases en inglés `x`, el modelo **muestrea** 4 traducciones al idioma objetivo
   (`grpo_generate_sequences`, T=1.8, top-k 100, top-p 0.95, sin gradiente).
2. Cada traducción se vuelve a traducir a inglés, 4 muestras por traducción → 2×4×4 = **32 secuencias** de vuelta por paso.
3. Recompensa = chrF++ (sacrebleu, `word_order=2, char_order=6`) de cada reconstrucción frente a `x`
   (`utils.py::grpo_compute_loss_and_logs`; la mezcla con BLEU está comentada, solo chrF++).
4. Ventaja = recompensa estandarizada dentro de cada grupo de 4; pérdida GRPO con clipping (0.2) y KL (β=0.04)
   frente a un modelo de referencia que se refresca cada 16 pasos de optimizador.
5. **Solo se retropropaga la dirección de vuelta (objetivo → inglés).** La dirección de ida solo se muestrea; su
   chrF contra la referencia real se calcula únicamente como métrica de log (`train/forward_chrf`).
   La mejora de eng→tgt (lo que mide el paper) es, por tanto, **indirecta**: el mismo modelo mejora entendiendo el
   idioma objetivo.
6. Evaluación = chrF++ eng→tgt en test (greedy, `num_beams=1`, `max_new_tokens=63`) antes y después de entrenar.

Datos: `breakend/nllb-multi-domain`, config `eng_Latn-<lang>`; 6000 train / 1310 valid / ~1600 test; 3 dominios
(noticias, conversación, salud). 6 idiomas: `ayr_Latn fur_Latn wol_Latn bho_Deva dyu_Latn rus_Cyrl`.

## 1.2 Qué dice el paper que hay que obtener

Tabla 1 (chrF++ en test, eng→tgt; extraída de la versión HTML de arXiv, **verificar contra el PDF** antes de citar):

| 600M | Base | BT | UMT | RT | **Ours** | Δ Ours−Base |
|---|---|---|---|---|---|---|
| Aimara `ayr` | 24.38 | 27.34 | 24.85 | 26.68 | **28.13** | +3.75 |
| Wolof `wol` | 20.73 | 23.57 | 19.89 | 22.84 | **23.96** | +3.23 |
| Ruso `rus` | 48.80 | 53.31 | 51.70 | 51.63 | 52.79 | +3.99 |
| Friulano `fur` | 45.75 | 47.38 | 48.96 | 46.25 | 47.61 | +1.86 |
| Dyula `dyu` | 18.03 | 20.13 | 20.82 | 21.89 | **22.50** | +4.47 |
| Bhojpuri `bho` | 43.07 | 45.36 | 39.79 | 41.62 | 43.97 | +0.90 |
| **Media** | 34.91 | 36.18 | 34.34 | 35.15 | **36.49** | +1.58 |

(la tabla de 1.3B está en `scripts/report.py`). Hiperparámetros del paper: AdamW, lr 2e-6, **2 épocas**, batch 2,
K=4, β=0.04, ε=0.2, referencia cada 16 pasos, T=1.8/top-k 100/top-p 0.95. **No reporta**: hardware, tiempo,
semillas, longitud máxima. Dice "unos 8K pasos" pero 2 épocas × 6000 / batch 2 = 6000 pasos (no lo concilia).

Lecturas importantes de esta tabla para diseñar la réplica:

* Las mejoras son **pequeñas** (media +1.6 chrF++, ruido de una sola semilla) y **RT** (el mismo round-trip pero con
  entropía cruzada, sin RL) consigue casi lo mismo en varios idiomas (dyu: +3.86 frente a +4.47). Es decir, buena parte
  de la ganancia **no depende del RL**, y el efecto RL-vs-MLE que sí es específico del método es de ~0.4–1.5 puntos.
* Los idiomas con más margen y mejor señal/ruido son **dyu, ayr y wol** (+3 a +4.5 con 600M). Ruso/friulano/bhojpuri
  tienen el Base alto y BT (que necesita texto en el idioma objetivo) gana. → los pilotos van con **ayr y dyu**, 600M.

## 1.3 Hallazgos (por qué es fácil "no ver nada" al replicar)

Verificados en el código salvo que se indique.

| # | Hallazgo | Consecuencia |
|---|---|---|
| H1 | `configs/task/nllb_md.yaml` original apunta a `model.name=/workspace/thesis/.../epoch_0_rus_Cyrl_chrf` (checkpoint local del autor tras **1 época**) y `epochs: 1`. El autor encadenaba ejecuciones de 1 época; el config no es el "run desde cero". | Para replicar hay que partir del modelo público y usar `epochs=2` (o encadenar). Es el único cambio *obligatorio* del config. |
| H2 | `eval.every_n_opt_steps: 5`: cada 5 pasos se evalúa **todo el set de validación** (1310 frases). En 6000 pasos son **1200 evaluaciones** de 1310 frases. | Con el config por defecto la ejecución completa es prácticamente inviable en una 2080; probablemente es la razón de que "no termine". Se sustituye por validación en un subconjunto cada N pasos (solo monitorización, no afecta al entrenamiento). |
| H3 | Memoria: NLLB-600M tiene 615M parámetros, de los cuales **262M son el embedding compartido** (vocabulario de 256k). Entrenar todo con AdamW fp32 ≈ 2.5 GB pesos + 2.5 GB gradientes + 4.9 GB Adam ≈ **9.8 GB**, más logits (n_seq × L × 256k × 4 B ≈ 2 GB para las 32 secuencias, ×2–3 con log_softmax/gradiente) y un 2º modelo de referencia. *(Estimación mía, no medida.)* | No cabe en una 2080 (8 GB) ni en una 2080 Ti (11 GB) tal cual. Hay perfiles `--mem` opt-in (ver 04_CAMBIOS.md). El paper no dice qué GPU usó. |
| H4 | Un smoke test con 4–8 pasos **no puede mostrar mejora**: con lr 2e-6 AdamW mueve cada peso ≈2e-6 por paso; los resultados de `feat/try` (8 pasos) dieron exactamente el mismo chrF++ (31.4217…) antes y después. | El smoke solo valida la tubería. Para ver señal hay que llegar a cientos/miles de pasos, de ahí el piloto de 600 pasos y la variante con lr mayor. |
| H5 | **Verificado:** el dataset está ordenado por dominio (en train, filas 0–~2000 `chat`, luego `news`, luego `health`). Un subconjunto `select(range(N))` (lo que hacía `feat/try` en su modo rápido) contiene **solo `chat`**. | Los subconjuntos ahora son aleatorios con semilla fija (`subset_seed`), idénticos antes y después. |
| H6 | El modelo se queda en `eval()` durante todo el entrenamiento (`from_pretrained` lo deja así y `_run_evaluation` solo restaura `train()` si estaba en train): **dropout desactivado**. | No es un bug por sí mismo (sampling y logprobs coherentes), pero hay que saberlo; no se toca. |
| H7 | Con eval-only, el test se evaluaba dos veces y `step_idx` se usaba sin definir al final. | Corregido (neutral). |
| H8 | El efecto esperado es pequeño y hay una sola semilla en el paper: con 1600 frases de test, ±0.5 chrF++ puede ser ruido. | Se usan 3 semillas cuando se pueda y **bootstrap pareado** antes/después (`scripts/report.py`). |
| H9 | Código/dep. rotos en un entorno actual: `torchdata.datapipes` (eliminado en torchdata ≥0.10) y `pytorch_lightning`; `requirements.txt` sin versiones; `datasets<4` necesario por `trust_remote_code`; `apply_lora` no definido (solo con `use_lora=true`, no usado en el paper). | `dl.py` con *shims* neutrales; versiones fijadas en `slurm/requirements-server.txt`. |

| H10 | **Verificado:** el script de carga de `breakend/nllb-multi-domain` descarga los datos de enlaces `tinyurl.com` (uno por dominio) en cada carga y, aunque exista caché, **falla en modo offline** (`OfflineModeIsEnabled: Tried to reach https://tinyurl.com/NLLBMDchat`). Durante mis pruebas el enlace funcionó una vez y más tarde dejó de responder desde mi red. | Depender de ese enlace en cada job de Slurm es frágil. `scripts/prefetch.py` guarda cada config con `save_to_disk` en `data/nllb-md/` y `dl.py` la usa si existe (sin red ni tinyurl). |
| H11 | Tamaños reales: train 6000, valid **1309**, test **1500** (el paper dice 1310 / 1600). | Las cifras absolutas del test no son idénticas al paper; se compara el delta Base→Ours. |

## 1.4 Qué NO sabemos / no podemos garantizar

* Si el "Base" del paper se midió con greedy o con beam search (el código usa greedy por defecto). Se comprueba en la
  etapa `base`: si nuestro Base difiere mucho del de la Tabla 1, probar `+task.eval.num_beams=4`.
* Tiempo por paso en las 2080: se mide en el smoke.
* Los números de la Tabla 1 los extraje de la versión HTML con una herramienta automática: contrastarlos con el PDF.
