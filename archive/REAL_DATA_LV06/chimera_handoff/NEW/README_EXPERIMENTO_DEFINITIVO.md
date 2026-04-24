# ASIC-RAG-HEALTH: Experimento Definitivo

## Diferencia crítica con experimentos anteriores

| Experimento anterior | Experimento definitivo |
|---------------------|------------------------|
| Job estático: `"0"*64` | Job real: `merkle_root` de registros |
| ASIC mina independiente | ASIC mina datos médicos reales |
| Sin vínculo criptográfico | Prueba verificable por registro |

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUJO CRIPTOGRÁFICO                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. GENERACIÓN          2. MERKLE TREE        3. ASIC PROOF    │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐   │
│  │ Registro 1  │──┐    │             │       │             │   │
│  │ (cifrado)   │  │    │   Hash A    │──┐    │  Merkle     │   │
│  └─────────────┘  ├───▶│             │  │    │  Root       │   │
│  ┌─────────────┐  │    └─────────────┘  │    │      +      │   │
│  │ Registro 2  │──┤    ┌─────────────┐  ├───▶│  Nonce      │──▶│ PRUEBA
│  │ (cifrado)   │  │    │   Hash B    │──┘    │  (ASIC)     │   │ VERIFICABLE
│  └─────────────┘  ├───▶│             │       │             │   │
│  ┌─────────────┐  │    └─────────────┘       └─────────────┘   │
│  │ Registro N  │──┘                                             │
│  │ (cifrado)   │                                                │
│  └─────────────┘                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Uso

### Paso 1: Verificar lógica criptográfica (sin hardware)
```bash
python verify_crypto_binding.py
```

### Paso 2: Configurar LV06
1. Conectar LV06 a la misma red WiFi
2. Configurar Stratum pool:
   - URL: `stratum+tcp://TU_IP_PC:3333`
   - Worker: `clinic_node`
   - Password: `x`

### Paso 3: Ejecutar experimento
```bash
python DEFINITIVE_ASIC_RAG_EXPERIMENT.py --duration 300 --patients 50
```

## Métricas capturadas

1. **Vinculación criptográfica**
   - Registros procesados
   - Merkle roots sellados
   - Pruebas ASIC generadas
   - Tasa de verificación

2. **Rendimiento**
   - Hashrate promedio (MH/s)
   - Hashrate pico
   - Total de shares

3. **Latencia**
   - Media, P50, P95, P99
   - Desviación estándar
   - Coeficiente de variación (entropía física)

4. **Energía**
   - Consumo (W)
   - Eficiencia (MH/W)
   - Comparativa vs GPU

## Output

El experimento genera:
- `definitive_results_TIMESTAMP.json` - Datos completos
- Informe en consola con todas las métricas
- Verificación de pruebas criptográficas

## Validación científica

Cada prueba ASIC contiene:
```json
{
  "merkle_root": "hash de los registros médicos",
  "nonce": "valor encontrado por el ASIC",
  "block_hash": "SHA256(merkle_root + nonce)",
  "records_count": 5,
  "timestamp": 1234567890.123
}
```

La prueba es verificable: cualquiera puede recalcular `SHA256(merkle_root + nonce)` y confirmar que el ASIC procesó esos datos específicos.
