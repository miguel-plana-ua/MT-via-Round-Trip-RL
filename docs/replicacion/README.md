# Réplica de "Improving Low-Resource MT via Round-Trip RL"

Rama de trabajo: `feat/replication` (creada desde la rama original `src-tgt-src-outcome`).

| Documento | Contenido |
|---|---|
| [01_ANALISIS.md](01_ANALISIS.md) | Cómo funciona el código, qué dice el paper, hallazgos que explican por qué cuesta replicar |
| [02_PLAN.md](02_PLAN.md) | Plan por etapas (smoke → base → piloto → completo → ampliar) con criterios de éxito |
| [03_SERVIDOR.md](03_SERVIDOR.md) | Comandos paso a paso para Slurm + miniconda con 2 GPUs |
| [04_CAMBIOS.md](04_CAMBIOS.md) | Cada cambio respecto al original, por qué, y si altera o no el cómputo |
| [DIARIO.md](DIARIO.md) | Registro cronológico de lo hecho, lo verificado y lo **no** verificado |

## Resumen en 10 líneas
1. El método entrena **solo la dirección de vuelta** (tgt→eng) con GRPO y recompensa chrF++ de la reconstrucción; la mejora eng→tgt es indirecta.
2. Mejoras del paper: +0.9 a +4.5 chrF++ (media +1.6, 600M), una semilla; el control RT (MLE) logra casi lo mismo en varios idiomas.
3. El config original apunta a un checkpoint local del autor y usa `epochs=1`; hay que partir del modelo público.
4. `eval.every_n_opt_steps=5` implica ~1200 evaluaciones completas de validación: inviable; se sustituye por validación en subconjunto.
5. Entrenar 600M completo con AdamW fp32 no cabe en una 2080 (8 GB) / 2080 Ti (11 GB): hay perfiles de memoria opt-in.
6. Una prueba de 8 pasos no puede mostrar mejora: sirve solo para validar la tubería.
7. Etapa 2 (`base`) valida datos/métrica sin entrenar: el Base debe parecerse al de la Tabla 1.
8. Pilotos en `ayr_Latn`/`dyu_Latn` (mayor ganancia en el paper), 600M, con bootstrap pareado antes/después.
9. Una ejecución usa las 2 GPUs reservadas (política en `cuda:1`, referencia en `cuda:0`).
10. Todo lo anterior se ha ejecutado ya en las 2080 Ti de `abaco`: ver [DIARIO.md](DIARIO.md) para lo probado y los resultados.

## Estado de la réplica (23-oct): resultados con NLLB-600M, test de 1500 frases, semilla 27, `--mem lowmem1`
| Idioma | Base | Ours (nuestro) | Ours (paper) | Δ nuestro | Δ paper |
|---|---|---|---|---|---|
| `ayr_Latn` | 24.38 | 28.08 | 28.13 | +3.69 | +3.75 |
| `dyu_Latn` | 18.03 | 22.62 | 22.50 | +4.58 | +4.47 |

Pendiente: `wol`, `fur`, `bho`, `rus` completos, más semillas, control RT y 1.3B (ver `02_PLAN.md`). Detalle en `DIARIO.md`.

