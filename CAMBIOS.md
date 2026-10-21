# Registro de cambios (CHANGELOG de la réplica)

Objetivo: poder reproducir todos los resultados del paper (métodos Ours / BT / RT / UMT × NLLB 600M y 1.3B × 6 idiomas) con un entorno reproducible gestionado con `uv`. Los hiperparámetros del paper se han contrastado con <https://arxiv.org/html/2601.12535>.

## 1. Entorno con uv

| Archivo | Cambio | Motivo |
|---|---|---|
| `pyproject.toml` (nuevo) | Dependencias con versiones fijadas (`torch==2.8.0`, `transformers==4.57.1`, `datasets==3.6.0`, `sacrebleu==2.5.1`, `hydra-core==1.3.2`, `wandb`, `sentencepiece`, etc.), Python `>=3.11,<3.13`, `package = false`. | Reproducibilidad. `datasets<4` es necesario porque `breakend/nllb-multi-domain` usa un script de carga (`trust_remote_code`). |
| `uv.lock` (nuevo, generado) | Lockfile con 83 paquetes resueltos. | Versiones exactas idénticas en el portátil y en el PC con GPU. |
| `requirements.txt` (eliminado) | Sustituido por `pyproject.toml`. | Sin versiones fijadas y con paquetes innecesarios o rotos (`torchdata`, `lightning`, `spacy`, `loralib`...). |

Uso: `uv sync` y luego `uv run python ...`. En el PC con GPU puede ser necesario instalar el build de torch con CUDA que corresponda (`--index-url` de PyTorch) si el de PyPI no lo trae.

## 2. Código principal

### `dl.py`
- Quitado `torchdata` (`MapDataPipe`, eliminado en versiones recientes) y `pytorch_lightning`. `TranslationDataModule` ya no hereda de `LightningDataModule` (usaba solo `save_hyperparameters`, sustituido por un `SimpleNamespace` en `self.hparams`) y `TranslationDataPipe` ahora es un `torch.utils.data.Dataset`. **Motivo:** eliminar dependencias frágiles; el comportamiento es el mismo.
- Nuevos parámetros `max_train_samples` y `max_eval_samples` en `TranslationDataModule` y en `_resolve_split(split_names, limit)`. **Motivo:** poder hacer pruebas rápidas (smoke) en CPU. Con `null` (por defecto) se usa el dataset completo.

### `main.py`
- `_run_evaluation`: al terminar cada evaluación se añade una línea a `<output_dir>/metrics.jsonl` (`split`, `step`, `<split>/chrf++`). **Motivo:** tener los resultados en un fichero parseable sin depender de wandb (lo usa `scripts/collect_results.py`).
- La ruta usa `HydraConfig.get().runtime.output_dir` (import añadido `from hydra.core.hydra_config import HydraConfig`, y `import json`). **Motivo:** Hydra 1.3 no cambia el directorio de trabajo, y una primera versión con ruta relativa escribía `metrics.jsonl` en la raíz del repo; detectado en la prueba y corregido.
- Guardado del modelo por época: ahora va a `<output_dir>/model/...` (antes `os.getcwd()/model`) y se puede desactivar con `task.training.save_model=false`. **Motivo:** cada experimento guarda en su carpeta, sin pisarse, y se evita llenar disco en pruebas.
- `step_idx = 0` inicializado antes del bucle de entrenamiento. **Motivo:** con `eval.only=true` la evaluación final usaba `step_idx` sin definir (bug latente).
- Eliminado `attn_implementation="flash_attention_2"` al cargar el modelo de referencia (solo si hay `reference_name`). **Motivo:** requiere el paquete `flash-attn` y GPU; no es necesario para replicar.
- Pasa `max_train_samples` / `max_eval_samples` desde `config.task.data` al data module.

### Sin cambios en la lógica científica
`utils.py` (pérdida GRPO, recompensa chrF), el bucle de entrenamiento y la evaluación no se han tocado.

## 3. Configuración

### `configs/task/nllb_md.yaml`
- `model.name`: de una ruta local del autor (`/workspace/thesis/...epoch_0...`) a `facebook/nllb-200-distilled-600M` (1.3B comentado). **Motivo:** la ruta no existe fuera de su máquina y además era un checkpoint ya entrenado.
- Eliminada la línea comentada del 3.3B (el paper solo usa 600M y 1.3B distilled).
- `training.epochs`: `1` → `2`. **Motivo:** el paper indica 2 épocas.
- Añadidos `data.max_train_samples: null`, `data.max_eval_samples: null` y `training.save_model: true`. **Motivo:** soporte de las pruebas rápidas y del guardado opcional descritos arriba.
- Resto de hiperparámetros sin cambiar; ya coinciden con el paper (lr 2e-6, batch 2, K=4, β=0.04, clip 0.2, T=1.8, top-k 100, top-p 0.95, refresco de referencia cada 16 pasos).

## 4. Baselines

### `baselines/backtranslation/backtranslation.py`, `baselines/roundtrip-bt/roundtrip-bt.py`, `baselines/UMNMT/umnmt.py`
- Nuevo argumento `--model_name` y uso en `main` en lugar del modelo fijado en el código. Por defecto conserva el valor anterior (BT y RT: 1.3B; UMT: 600M). **Motivo:** el modelo estaba fijado y era inconsistente entre scripts; ahora se puede lanzar 600M y 1.3B para cada baseline.

### `run_backtranslation.sh`, `run_roundtrip_bt.sh`, `run_training.sh`
- `/workspace/miniconda3/envs/gfn/bin/python` → `python`. **Motivo:** ruta del entorno del autor. Se ejecutan con `uv run bash ...` o con el venv activado.
- Nota: estos scripts siguen pasando solo los argumentos originales (no `--model_name`); el lanzador recomendado es `scripts/run_all.py`.

## 5. Scripts nuevos

### `scripts/run_all.py`
Lanzador de todos los experimentos: `--methods {ours,bt,rt,umt}`, `--models {600M,1.3B}`, `--langs` (por defecto ayr_Latn, fur_Latn, wol_Latn, bho_Deva, dyu_Latn, rus_Cyrl), `--smoke` (4 muestras, 1 época), `--dry-run`, `--no-wandb`. Guarda cada ejecución en `results/<método>/<modelo>/<idioma>/` con `log.txt` y un marcador `DONE`, y salta las ya terminadas para poder reanudar. **Motivo:** una única orden reproducible para toda la matriz de resultados. Hiperparámetros de los baselines según el paper (BT/RT: 2 épocas, batch 2, lr 3e-5; UMT: lr por defecto 3e-5 / 5e-5 en disc.).

### `scripts/collect_results.py`
Lee `metrics.jsonl` (Ours y Base = evaluación en test antes del entrenamiento) y los `log.txt` de los baselines (`chrF++ (src→tgt)`) y genera tablas markdown tipo Tabla 1 del paper para 600M y 1.3B. **Motivo:** agregar resultados sin hacerlo a mano.

## 6. Estado de la validación
- `uv lock`/`uv sync` correctos; `run_all.py --dry-run` genera bien los comandos; sintaxis de todos los `.py` editados verificada.
- Prueba real en CPU (600M, ruso, 4 frases): carga de dataset y modelo y evaluación inicial funcionan (chrF++ test 32.30, val 36.19 sobre 4 frases, solo orientativo). **El entrenamiento no llegó a completarse en CPU** (la prueba se interrumpió por lentitud: cada evaluación de 4 frases tardó ~1 min), así que el bucle GRPO y los baselines **no están validados** aún. Hay que verificarlo en la máquina con GPU con `--smoke` antes de lanzar todo.
- Tras la interrupción se borraron `results/smoke_*` y el `metrics.jsonl` mal ubicado de la raíz.

## 7. Pendientes conocidos
- `main.py` usa `apply_lora` sin importarla (solo afecta con `use_lora=true`; no se usa en el paper).
- La métrica de entrenamiento mezcla chrF y BLEU en el paper, pero el código solo usa chrF++ como recompensa (BLEU comentado en `utils.py`); no se ha modificado.
- El paper también reporta fluidez con modelos GoldFish (Tabla 2), COMET para ruso y curvas de validación; **no hay código para ello en el repo** y habría que añadirlo si quieres replicar esas partes.
- Los 22 de "epochs 22 / batch 22" del texto del paper para UMT parecen un fallo de extracción (probablemente 2); se mantuvo 2.
- `uv.lock` se resolvió en Windows; en Linux con GPU puede reajustarse el build de torch/CUDA.
