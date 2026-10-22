# 2. Plan de réplica por etapas

Principio: cada etapa responde **una** pregunta, cuesta más que la anterior y solo se pasa a la siguiente si la
anterior sale bien. Todo se lanza con `scripts/run_stage.py` (en Slurm con `slurm/submit.sh`, ver 03_SERVIDOR.md);
los resultados quedan en `results/<etapa>/<modelo>/<idioma>/seed<N>/` y se resumen con `scripts/report.py`.

Hardware asumido: 2 GPUs 2080 (modelo 600M; `main.py` pone la política en `cuda:1` y la referencia en `cuda:0`,
así que **una ejecución usa las 2 GPUs reservadas**). El 1.3B queda fuera hasta que el 600M funcione.

| Etapa | Pregunta | Coste (estimado) | Criterio de éxito |
|---|---|---|---|
| 0. Entorno | ¿Corre torch con CUDA y las versiones fijadas? ¿Qué GPU son exactamente? | minutos | `check_env.py` imprime 2 GPUs y su memoria |
| 1. `smoke` | ¿La tubería completa (generación → recompensa → backward → evaluación) corre y cabe en memoria? ¿Cuántos s/paso? | ~10–20 min | Termina; `peak_mem_gb` < memoria de la GPU; se anota `s/paso` |
| 2. `base` | ¿Nuestro Base chrF++ coincide con el de la Tabla 1? (valida datos, tokenización, métrica y decodificación **sin entrenar nada**) | ~5–15 min por idioma | Base dentro de ~±1 punto del paper |
| 3. `pilot` | ¿El entrenamiento hace lo que debe (recompensa de vuelta sube, KL pequeño, forward chrF sube) y mejora el test? | 600 pasos ≈ `600 × s/paso` + evaluaciones | Sin degradación; tendencia positiva (ver abajo) |
| 4. `full` | ¿Reproducimos Base→Ours del paper en 1 idioma (600M)? | 6000 pasos ≈ `6000 × s/paso` (horas) | Δ test del mismo signo y orden de magnitud que el paper, bootstrap significativo |
| 5. Ampliar | Más idiomas, 3 semillas, baseline RT (control MLE), después 1.3B | días | Tabla tipo Tabla 1 reducida |

## Detalle por etapa

### Etapa 0: entorno
```bash
bash slurm/setup_env.sh
python scripts/check_env.py --load-model      # mejor dentro de un job con GPU, ver 03_SERVIDOR.md
```
**Dato que decide el perfil de memoria:** modelo exacto y memoria de las GPUs (2080 = 8 GB, 2080 Ti = 11 GB).

### Etapa 1: smoke (40 frases, 20 pasos —cruza el refresco de la referencia del paso 16—, val/test de 32 frases)
```bash
slurm/submit.sh smoke --lang ayr_Latn --mem lowmem
```
Qué mirar en `python scripts/report.py results/smoke/600M/ayr_Latn/seed27 --no-bootstrap`:
`peak GPU memory`, `s/step`, que existan `train_log.jsonl` y `hyps/`. **No** se espera mejora (hallazgo H4).
Si da OOM con `lowmem` en 8 GB, repetir con `--mem tight` (requiere `pip install bitsandbytes`).

### Etapa 2: Base en los 6 idiomas (solo evaluación)
```bash
for l in ayr_Latn dyu_Latn wol_Latn rus_Cyrl fur_Latn bho_Deva; do slurm/submit.sh base --lang $l; done
python scripts/report.py --table results
```
Compara con la columna Base 600M (24.38, 18.03, 20.73, 48.80, 45.75, 43.07). Nota: aquí el test tiene 1500 frases
y valid 1309 (el paper habla de 1600/1310), así que no se espera coincidencia exacta. Si el Base queda muy lejos
(>2 puntos), probar `--set +task.eval.num_beams=4` y revisar tokenización/`max_new_tokens`. **No pasar a entrenar
hasta que el Base sea razonable**: si no, cualquier delta posterior no es comparable con el paper.

### Etapa 3: piloto (600 pasos)
Dos variantes, ambas con `ayr_Latn` (y luego `dyu_Latn`):
```bash
slurm/submit.sh pilot --lang ayr_Latn --tag lr2e-6                    # hiperparámetros del paper
slurm/submit.sh pilot --lang ayr_Latn --tag lr1e-5 --lr 1e-5          # DESVIACIÓN: lr ×5 para ver señal antes
```
Qué se espera y cómo interpretarlo (honestamente, sin conocer las curvas del autor):
* Con 600 pasos y lr 2e-6 solo se recorre el 10 % del entrenamiento; la ganancia final del paper (+3.75 en ayr) se
  alcanza tras ~6000 pasos, así que aquí puede verse **poco o nada** en el test. Lo que sí debe verse: `backward reward`
  (la recompensa que optimiza el RL) subiendo entre el primer y el último 10 % de pasos, KL pequeño (≪1) y estable, y el
  `forward chrF` (contra la referencia real, solo monitorización) con tendencia ≥ 0.
* La variante lr 1e-5 es una prueba de **mecanismo**, no de réplica: si con ella el test sube, el método funciona en tu
  tubería; si sube la recompensa de vuelta pero el test baja, hay *reward hacking* o un error en la tubería.
* Reglas de decisión: (a) reward sube y test sube → pasar a `full`; (b) reward sube, test igual → pasar a `full` igualmente
  (el piloto es corto); (c) reward no sube → revisar antes de gastar horas (lr, muestreo, bug); (d) NaN/KL explota →
  parar y revisar.

### Etapa 4: ejecución completa de 1 idioma
```bash
TIME=24:00:00 slurm/submit.sh full --lang ayr_Latn --mem lowmem --tag run1
```
2 épocas × 3000 batches = 6000 pasos, validación en 300 frases cada 500 pasos, test completo antes y después.
**Aviso de fidelidad:** `--mem lowmem/tight` congelan embeddings (y `tight` cambia el optimizador): son desviaciones. Si
algún día tienes una GPU de ≥24 GB, usa `--mem paper`. La comparación con el paper se hace siempre sobre el delta
Base→Ours del *mismo* modelo medido por nosotros, no sobre el valor absoluto.

### Etapa 5: ampliar (solo si la 4 funciona)
1. `dyu_Latn` y `wol_Latn` (los otros con mayor ganancia en el paper); 3 semillas (una ejecución por semilla: `--seed 27`, `--seed 28`, `--seed 29`).
2. Baseline **RT** (`baselines/roundtrip-bt`) en el mismo idioma: es el control que dice cuánto aporta el RL frente al
   MLE. Requiere ajustar memoria (los baselines entrenan todo con AdamW fp32): pendiente de la etapa 4.
3. 1.3B: pesos 5.5 GB en fp32; necesitaría `tight` + más cambios o GPUs mayores. No planificado todavía.

## Presupuesto de tiempo (fórmula, rellenar con el `s/paso` del smoke)
`horas_entrenamiento ≈ pasos × s_por_paso / 3600`: piloto 600 pasos; completo 6000 pasos. Añade las evaluaciones:
test completo (1500 frases, greedy, batch 48) antes y después, y validación periódica de 300 frases.
