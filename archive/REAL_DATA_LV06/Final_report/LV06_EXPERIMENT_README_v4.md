# LV06 Definitive Capability Assessment

## Qué hace este experimento

Este script determina **honestamente** qué puede y qué no puede hacer el Lucky Miner LV06 como sustrato computacional. No asume nada — prueba todo.

## Metodología

### Rate-Encoded Reservoir Computing (RE-RC)

```
Input u[t] ──► Difficulty D = D_base/(u+ε) ──► Share Rate λ ∝ 1/D ──► Features ──► Ridge
```

La modulación de difficulty es la única forma de crear **acoplamiento físico** real entre input y hardware.

### Batería de Tests

| Fase | Test | Qué Mide | Criterio de Éxito |
|------|------|----------|-------------------|
| 1 | NARMA-10 Normal | Capacidad predictiva | NRMSE < 0.20 |
| 1 | NARMA-10 Shuffle | Control de causalidad | NRMSE > Normal + 10% |
| 1 | NARMA-10 Constant | Control de entropía | NRMSE > Normal + 10% |
| 2 | Memory Capacity | Memoria temporal | MC > 0.5 steps |
| 3 | XOR Temporal | No-linealidad | Accuracy > 60% |
| 4 | Poisson Baseline | Supera teoría | Hardware < Poisson - 5% |

### Verdicts

El script emite verdicts claros:

- **✅ GENUINE RESERVOIR COMPUTER**: Cumple todos los criterios de RC
- **⚠️ LINEAR RESERVOIR**: RC sin no-linealidad demostrada
- **⚠️ RATE ENCODER**: Acoplamiento físico pero no RC completo
- **❌ NO RC CAPABILITY**: No funciona como reservoir

## Uso

### Requisitos

```bash
# Solo Python 3.8+ estándar - SIN dependencias externas
python3 lv06_definitive_experiment.py
```

### Configuración del LV06

1. Accede a la interfaz web del LV06
2. Ve a **Miner Configuration**
3. Configura Pool 1: `stratum+tcp://TU_IP:3333`
4. Usuario: `worker1`, Password: `x`
5. Guarda y aplica

### Ejecución

```bash
python3 lv06_definitive_experiment.py
```

El experimento tarda ~30 minutos (ajustable vía `ExperimentConfig`).

## Qué esperar (predicción honesta)

Basado en la física del sistema:

### Probablemente PASARÁ ✅

- **Rate Encoding Coupling**: La modulación de difficulty SÍ cambia el share rate
- **Entropía**: El mining es estocástico (proceso Poisson)

### Probablemente FALLARÁ ❌

- **Beats Poisson Baseline**: Es probable que un Poisson simulado iguale o supere al hardware
- **Memory Capacity > 0.5**: El hardware no tiene "memoria" intrínseca
- **XOR Nonlinearity**: El mapping rate→shares es mayormente lineal

### Diagnóstico

Si **Normal ≈ Shuffle**: No hay causalidad input→output
Si **Normal ≈ Constant**: La entropía no aporta información útil
Si **Hardware ≈ Poisson**: El hardware es "solo" un proceso Poisson modulado
Si **XOR ≈ 50%**: No hay procesamiento no-lineal

## Outputs

```
lv06_definitive_report_YYYYMMDD_HHMMSS.json  # Datos completos
lv06_definitive_report_YYYYMMDD_HHMMSS.md    # Resumen legible
```

## Filosofía

> "Es mejor tener un resultado honesto de 'no funciona' que un paper publicado con claims inflados."

Este experimento está diseñado para resistir peer review. Si pasa, tienes evidencia sólida. Si falla, sabes exactamente por qué y puedes pivotar.

## Personalización

Edita `ExperimentConfig` al inicio del script:

```python
@dataclass
class ExperimentConfig:
    NARMA_STEPS: int = 500        # Más pasos = más significancia estadística
    WINDOW_TIME: float = 1.0      # Segundos por paso
    D_BASE: float = 25.0          # Difficulty base (ajustar según hash rate)
    MIN_IMPROVEMENT_PERCENT: float = 10.0  # Umbral para "mejora significativa"
```

## Créditos

Fran (Agnuxo) - Concepto y hardware
Claude - Diseño experimental y código

Diciembre 2025
