# BlastingPy

Librería abierta de Python, con núcleo en C++20, para analizar los efectos de la voladura en minería: vibración del
terreno, sobrepresión de aire, proyección de rocas y vibración en campo cercano.

El caso de uso central: al terminar una campaña de monitoreo, el ingeniero de voladura tiene una tabla de eventos
(distancia, carga máxima por retardo, PPV, frecuencia). BlastingPy ajusta la ley de atenuación del sitio, calcula la
carga máxima por retardo y verifica el resultado contra un criterio normativo, indicando la edición, la tabla, la
magnitud y el método de frecuencia que usó.

**Estado:** en desarrollo, sin publicar (versión 0.1.0). El sprint 1 de 5 está completo: tabla de eventos, ley de
atenuación, carga máxima y veredicto de cumplimiento. El paquete no está en PyPI y el repositorio es privado.

## Instalación (desde el repositorio)

```bash
python -m pip install ".[test,io,viz]"
python -m pytest tests/python -q
```

Para compilar desde el código fuente hacen falta un compilador de C++ y CMake. Extras: `io` (leer y escribir `.xlsx`),
`viz` (gráficos).

## Ejemplo

```python
import blastingpy as bp
from blastingpy import attenuation, criteria, plotting

events = bp.read_events("campana.csv", units="metric", ppv_kind="component_max",
                        columns={"distance_m": "Distancia", "charge_kg": "Carga", "ppv_mm_s": "PPV",
                                 "frequency_hz": "Frec"})
law = attenuation.fit(events, law="sqrt")                  # PPV = K (D / sqrt(W))^-beta
result = attenuation.max_charge(law, 300.0, 12.7)          # kg por retardo a 300 m, PPV <= 12,7 mm/s
print(result.value, result.warnings)
verdict = criteria.evaluate(events, "USBM-RI8507-1980", "appendix-b-drywall")
print(verdict["status"].value_counts())
plotting.plot_site_law(law, events)
plotting.plot_compliance(events, "USBM-RI8507-1980", "appendix-b-drywall")
```

`write_events_template("events.csv")` escribe la tabla en blanco. Hay una versión completa y ejecutable con datos
sintéticos en [examples/01-site-law-and-compliance](examples/01-site-law-and-compliance/) (su README está en inglés).

## Convenciones

- **Unidades.** Dentro de la librería todo es SI: m, kg, mm/s, Hz. `read_events(units="imperial")` (o un diccionario por
  magnitud) convierte al leer; `blastingpy.units` tiene `to_si` y `from_si`.
- **Cota superior de la ley del sitio.** `k_upper = k * 10^(z * sigma)` con `z = 2` por omisión, como en USBM RI 8507 y
  en la práctica de BIVDEP de OSMRE. Para la cota unilateral del 95 % usa
  `attenuation.fit(events, z=attenuation.normal_quantile(0.95))` (z = 1,645). `max_charge` y `predict_ppv` usan la cota
  superior por omisión; pasa `attenuation.Level.Median` para la mediana.
- **Avisos.** Un ajuste con menos de 30 eventos, con `beta <= 0`, o una predicción fuera del rango de distancia o de
  carga del ajuste traen avisos (`law.warnings`, `result.warnings`). Una ley con `beta <= 0` no se usa para predecir.
  Una ley con `sigma == 0` o `z == 0` (por ejemplo, `make_site_law` sin `sigma`) tiene la cota superior igual a la
  mediana: sus predicciones con la cota avisan que el resultado no tiene margen de seguridad.
- **Una magnitud por ley.** Una ley predice la magnitud de PPV con la que se ajustó (`ppv_kind`), así que solo se
  compara con límites de esa magnitud; `attenuation.fit` rechaza eventos que mezclan clases.

## Leyes del sitio

| `law` | Forma (SI) | Fuente | Marca |
|---|---|---|---|
| `"sqrt"` | PPV = K (D / W^0,5)^-beta | USBM Bulletin 656, ec. 4.20 y 6.1 | V1 |
| `"cbrt"` | PPV = K (D / W^(1/3))^-beta | USBM Bulletin 656, ec. 4.13 con alfa = 1/3 (allí figura como propuesta de otros autores) | V2 |
| `"general"` | PPV = K W^a D^-beta | USBM Bulletin 656, ecs. 4.10 a 4.13 | V1 |

## Perfiles

`criteria.list_profiles()` devuelve los ids. Marcas: **V1** valores leídos en la fuente primaria; **V2** valores leídos
en una fuente secundaria confiable, no en la norma.

| Perfil | Ids de criterio | Magnitud | Marca |
|---|---|---|---|
| `USBM-RI8507-1980` | `appendix-b-drywall`, `appendix-b-plaster`, `table-13-drywall`, `table-13-plaster` | componente máxima, curva de frecuencia | V1 |
| `OSMRE-30CFR816.67` | `distance-table` | componente máxima, tabla por distancia (distancia escalada) | V1 |
| `PE-MEM-1995` | `distance-table` | componente máxima, tabla por distancia (guía del MEM del Perú) | V1 |
| `BS-7385-2-1993` | `line-1`, `line-2` | componente máxima, curva de frecuencia | V2 |
| `DIN-4150-3-2016` | `line-2`, `line-3` (solo cimentación) | componente máxima, curva de frecuencia | V2 |

**Un perfil nunca reemplaza a la norma.** Es una transcripción de valores límite con su cita. Los dos perfiles V2
(BS 7385-2 y DIN 4150-3) se transcribieron de fuentes secundarias y hay que cotejarlos con la norma comprada antes de
apoyarse en ellos. Los valores de DIN 4150-3 se leyeron de una figura, en líneas de cuadrícula rotuladas.

Para ver las reservas de un perfil, que la tabla del veredicto solo muestra como marca:

```python
c = criteria.load_profile("DIN-4150-3-2016").criteria["line-2"]
print(c.verification, c.reference, c.note["es"], sep="\n")
```

## Qué significa el veredicto

`criteria.evaluate(events, profile, criterion)` devuelve una fila por evento con `status`, `limit_mm_s`, `margin_mm_s`,
`margin_pct`, `governing`, `verification` y `reason`. Los cuatro estados:

- `complies`: todos los valores candidatos se evaluaron y están en el límite o por debajo.
- `exceeds`: al menos un valor evaluado supera su límite (el `reason` lista lo que no se pudo evaluar).
- `not_comparable`: falta la magnitud que pide el perfil o está incompleta, un valor es dato inválido, o falta la
  frecuencia (o la distancia) que el criterio necesita.
- `indeterminate`: una frecuencia o una distancia cae fuera del dominio del criterio.

«Cumple» exige conocer por completo la magnitud que pide el perfil. En los perfiles de componente máxima eso son las
tres componentes (`ppv_t_mm_s`, `ppv_v_mm_s`, `ppv_l_mm_s`) o un pico declarado con `ppv_kind="component_max"`. Un PPV de
`vector_sum` nunca se compara con un límite de componente. Limitación aceptada: un pico `component_max` con su
frecuencia se toma como la componente máxima del evento; las componentes menores a frecuencias más bajas que el evento
no trae no se revisan.

`reason` queda vacío en `complies`, salvo cuando un veredicto contra una curva de frecuencia se apoya en ese pico sin
las tres componentes: entonces dice `peak-only check: components at other frequencies are not verified`. La tabla del
veredicto lleva en `verdict.attrs` contra qué se comparó (`profile_id`, `criterion`, `citation`, `edition`,
`quantity`, `frequency_methods`, `verification`, `reference`), y `criteria.candidates(events, profile, criterion)`
lista, por evento, los valores que el veredicto comparó con el límite (el gráfico de cumplimiento dibuja justo esos).

## Tu propio perfil

Un perfil propio sirve para aplicar un límite que tienes de una norma que no está incluida, o uno fijado por un permiso
o un contrato. Se pasa como diccionario (o la ruta a un JSON); sus criterios llevan la marca
`"user"`:

```python
mine = {"profile_id": "MY-LIMIT", "quantity": "component_max", "units": {"velocity": "mm/s"},
        "criteria": [{"id": "flat-10", "kind": "flat", "limit": 10.0, "reference": "Contract clause 4"}]}
criteria.evaluate(events, mine, "flat-10")
```

El `quantity` del perfil debe ser una magnitud que traigan los eventos; si no, todos los veredictos son `not_comparable` y
`reason` dice qué falta. Tipos de criterio: `flat` (el de arriba), `frequency_curve` y `distance_table`; los JSON que
viajan en `python/blastingpy/profiles/` ilustran los dos últimos. Todo límite, frecuencia y distancia de un perfil debe
ser un número finito > 0 (`f_min` puede ser 0); cualquier otra cosa se rechaza al cargar el perfil.

## Qué falta todavía

Las leyes Langefors-Kihlström, Ghosh-Daemen y CMRI, y las normas UNE 22381, AS 2187.2, BS 6472-2, SS 4604866,
SN 640312a y NBR 9653 no entran porque no se leyó ninguna fuente con sus fórmulas o valores. Tampoco la línea 1 de
DIN 4150-3 ni los criterios del último piso, la curva de la Figura 1 de OSMRE ni la línea 1 de BS 7385-2 bajo 4 Hz.
Detalle en [docs/CHANGELOG.md](docs/CHANGELOG.md).

## Licencia

MIT. Ver [LICENSE](LICENSE).
