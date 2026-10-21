# Explicación del repositorio y guía de replicación

Documento generado a partir de la lectura del código (`main.py`, `utils.py`, `dl.py`, `configs/`, `baselines/`). Paper: <https://arxiv.org/abs/2601.12535>.

## 1. Qué se hace

Se hace *fine-tuning* de un modelo NLLB (600M / 1.3B) para traducción de idiomas de pocos recursos **sin necesitar un modelo externo ni etiquetas adicionales**, mediante **aprendizaje por refuerzo (GRPO) con round-trip**:

1. Se toma una frase en inglés `x` (dataset NLLB-MD, par `eng_Latn → idioma_objetivo`).
2. El modelo la traduce al idioma objetivo muestreando `num_return_sequences=4` candidatas `y_1..y_4` (temperatura 1.8, top-k 100, top-p 0.95).
3. Cada candidata se traduce de vuelta a inglés, también con 4 muestras → 16 reconstrucciones `x'` por frase.
4. **Recompensa** = chrF++ (`word_order=2, char_order=6`, sacrebleu) entre `x'` y la frase original `x`. (El paper menciona chrF++ + BLEU, pero en el código el promedio con BLEU está **comentado**: `combined_scores_tensor = chrf_scores_tensor` en `utils.py`.)
5. **GRPO**: recompensas estandarizadas por grupo (media/desv. de las 4 reconstrucciones de cada candidata) como ventaja, ratio estilo PPO con *clipping* (`clip_param=0.2`) y penalización KL (`beta=0.04`) frente a un modelo de referencia congelado.
6. Importante: en `main.py` solo se calcula y retropropaga la pérdida del **paso de vuelta** (`loss_backward`, idioma objetivo → inglés). El paso de ida (inglés → objetivo) solo se muestrea (con `no_grad`) y se mide con chrF contra la referencia real **solo como métrica de log** (`train/forward_chrf`); no se usa como recompensa. Al ser el mismo modelo en ambas direcciones, mejora la comprensión del idioma objetivo y de ahí la traducción.
7. El modelo de referencia se refresca copiando la política cada `update_ref_policy_every_n_steps=16` pasos de optimizador.
8. Evaluación: chrF++ en validación (cada `every_n_opt_steps=5`, beam `num_beams` por defecto 1) y en test, antes y después del entrenamiento. Se registra todo en Weights & Biases. Se guarda el modelo por época en `./model/<modelo>_epoch_<n>_<lang>_chrf` (relativo al directorio de trabajo de Hydra, ver `configs/hydra/default.yaml`).

### Archivos
| Archivo | Función |
|---|---|
| `main.py` | Bucle completo: carga modelo, datos, evaluación, entrenamiento GRPO, wandb |
| `utils.py` | Generación por muestreo, log-probs por token, pérdida GRPO (recompensa, ventaja, KL) |
| `dl.py` | `TranslationDataModule`: carga `breakend/nllb-multi-domain` (HuggingFace) y crea dataloaders (val/test con batch 48) |
| `configs/task/nllb_md.yaml` | Hiperparámetros, modelo, idiomas |
| `configs/device/*.yaml` | Tipo de acelerador (gpu por defecto) |
| `baselines/` | Baselines: backtranslation, round-trip-BT y UMNMT (scripts `.sh` con argumentos por par de idiomas) |

### Hiperparámetros por defecto (`nllb_md.yaml`)
lr 2e-6, AdamW, batch 2, 4 secuencias por entrada, 1 época, `max_sentence_len` 63 tokens, seed 27, sin LoRA.

## 2. Cómo replicar los resultados

### 2.1 Requisitos
- GPU NVIDIA con CUDA (recomendado Linux). Si hay ≥2 GPUs, la política va en `cuda:1` y la referencia en `cuda:0`; con 1 GPU ambas van en la misma (el 600M cabe en ~16-24 GB aprox.). En CPU funcionaría, pero es inviable en la práctica.
- Cuenta en Weights & Biases (`wandb login`), o poner `task.training.use_wandb=false`.
- Acceso a internet para HuggingFace (modelo `facebook/nllb-200-distilled-600M` y dataset `breakend/nllb-multi-domain`).

### 2.2 Instalación
```bash
python -m venv .venv && source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Notas conocidas del código (detectadas al revisarlo):
- `requirements.txt` no fija versiones. Si hay errores, usa versiones compatibles: `dl.py` usa `torchdata.datapipes` (eliminado en torchdata ≥0.10, usa `torchdata<=0.9` con un `torch` compatible, p. ej. torch 2.3/2.4) y `pytorch_lightning` (instala `pytorch-lightning`, el paquete `lightning` solo no basta para ese import).
- `load_dataset(..., trust_remote_code=True)` exige `datasets<4.0` (p. ej. `datasets==3.x`).
- `main.py` usa `apply_lora` sin importarla: solo falla si activas `use_lora=true`; déjalo en `false` (valor por defecto).
- `flash_attention_2` solo se usa si defines `model.reference_name`; por defecto no se usa.

### 2.3 Configuración — paso obligatorio
En `configs/task/nllb_md.yaml` el `model.name` apunta a una ruta local del autor (`/workspace/thesis/...epoch_0...`), que es un checkpoint ya entrenado una época. Para partir desde cero cámbialo por:
```yaml
model:
  name: "facebook/nllb-200-distilled-600M"
```
Y elige el idioma (`data.target_lang` y `data.dataset_config_name`), p. ej.:
- `ayr_Latn` (aimara central), `fur_Latn` (friulano), `wol_Latn` (wolof), `rus_Cyrl` (ruso); el baseline también usa `bho_Deva` y `dyu_Latn`.
```yaml
data:
  target_lang: wol_Latn
  dataset_config_name: "eng_Latn-wol_Latn"
```
Se pueden pasar por línea de comandos (Hydra):
```bash
python main.py task.model.name=facebook/nllb-200-distilled-600M \
  task.data.target_lang=wol_Latn task.data.dataset_config_name=eng_Latn-wol_Latn
```
Si no usas wandb: `task.training.use_wandb=false`. **Ojo:** el código de evaluación llama a `wandb_run.log` solo si hay run, pero revisa que no falle sin wandb en tu versión.

Para el modelo 1.3B usa `facebook/nllb-200-1.3B` (o la variante distilled 1.3B), ya que el paper evalúa 600M y 1.3B.

### 2.4 Ejecución
```bash
python main.py
```
Flujo: evaluación inicial (test y val) → entrenamiento 1 época con evaluaciones en val cada 5 pasos → evaluación final (val y test). El **chrF++ de test antes vs. después** es el resultado a comparar con el paper (tablas/figuras). Hay que repetir por cada idioma (y tamaño de modelo); las épocas adicionales se consiguen reanudando desde el checkpoint guardado (así lo hace el autor con `epoch_0`) o subiendo `training.epochs`.

Para solo evaluar un modelo: `task.eval.only=true`.

### 2.5 Baselines
```bash
bash baselines/backtranslation/run_backtranslation.sh
bash baselines/roundtrip-bt/run_roundtrip_bt.sh
bash baselines/UMNMT/run_training.sh
```
Los `.sh` usan un intérprete fijo del autor (`/workspace/miniconda3/envs/gfn/bin/python`): sustitúyelo por `python`. Se lanzan por par de idiomas (lista `LANG_PAIRS`); edítala a los pares que quieras. Backtranslation usa `--num_epochs 2 --batch_size 2 --learning_rate 3e-5`.

### 2.6 Consejos para una réplica fiel
- Mantén seed 27 y los hiperparámetros por defecto; el muestreo es estocástico, así que espera pequeñas variaciones (conviene varias semillas).
- Compara con el modelo base (la evaluación en el paso 0 ya te da la línea base chrF++).
- Los gráficos del autor están en el informe de wandb enlazado en el `README.md`, útil para contrastar curvas.
- Registra versiones de librerías (`pip freeze`) para reproducibilidad.
