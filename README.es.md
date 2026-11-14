# BlastingPy

Librería abierta de Python, con núcleo en C++20, para analizar los efectos de la voladura en minería: vibración del
terreno, sobrepresión de aire, proyección de rocas y vibración en campo cercano.

El caso de uso central: al terminar una campaña de monitoreo, el ingeniero de voladura tiene una tabla de eventos
(distancia, carga máxima por retardo, PPV, frecuencia). BlastingPy ajusta la ley de atenuación del sitio, calcula la
carga máxima por retardo y verifica el resultado contra un criterio normativo, indicando la edición, la tabla, la
magnitud y el método de frecuencia que usó.

**Estado:** en desarrollo, sin publicar (versión 0.1.0). Los sprints 1 y 2 de 5 están completos: tabla de eventos, ley de
atenuación, carga máxima y veredicto de cumplimiento, y lectura de registros del sismógrafo (forma de onda, PPV, suma vectorial, frecuencia). El paquete no está en PyPI y el repositorio es privado.

## Instalación (desde el repositorio)

```bash
python -m pip install ".[test,io,viz]"
python -m pytest tests/python -q
```

Para compilar desde el código fuente hacen falta un compilador de C++ y CMake. Extras: `io` (leer y escribir `.xlsx`),
`viz` (gráficos), `seismic` (lectores de ObsPy).

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

## Desde un registro del sismógrafo

`blastingpy.io` lee la forma de onda que exporta un sismógrafo; `blastingpy.waveform` la convierte en picos, suma
vectorial y frecuencia, y en una fila de la tabla de eventos de las secciones anteriores. Los datos y su evidencia están
en [docs/investigacion/FUENTES_SPRINT2.md](docs/investigacion/FUENTES_SPRINT2.md).

```python
from blastingpy import attenuation, criteria, io, waveform

records = [io.read_instantel_ascii(p, sample_rate_hz=1024, pretrigger_s=0.25) for p in paths]
events = waveform.events_from_records(records, distance_m=[100, 150, 200], charge_kg=[20, 50, 30])
law = attenuation.fit(events, law="sqrt")
verdict = criteria.evaluate(events, "USBM-RI8507-1980", "appendix-b-drywall")
```

Una versión completa y ejecutable con datos sintéticos está en [examples/02-waveform-to-events](examples/02-waveform-to-events/).

| Lector | `format_evidence` | Notas |
|---|---|---|
| `read_instantel_ascii`, mínimo (`Tran Vert Long MicL` y filas) | `real-file` | Sin metadatos: pasa `sample_rate_hz` (y `pretrigger_s`). |
| `read_instantel_ascii`, THOR | `third-party-parser` | **No validado contra un archivo real**: construido desde la descripción de un lector de terceros. |
| `read_instantel_ascii`, Blastware con encabezado (tabla o bloques) | `manual-only` | **No validado contra un archivo real**: construido desde el manual; el registro lleva un aviso. Solo los separadores por omisión: encabezado entre comillas dobles y valores de los bloques separados por comas. Las unidades del encabezado son una pista: manda `units=`, con un aviso. |
| `read_instantel_histogram` | `manual-only` | **No validado contra un archivo real**: construido desde el manual; devuelve un DataFrame con un aviso en `attrs`. |
| `read_waveform_csv` | `user` | Cualquier marca o canales renombrados (`columns=`, `units=`, `sep="\t"`); acepta notación científica. |
| `read_obspy` (extra `seismic`) | `user` | miniSEED, SAC y otros formatos que lee ObsPy. |

Un export real de cada formato no validado lo resolvería. Mientras tanto, contrasta el primer registro que leas con lo
que muestra el software de tu sismógrafo.

- **La frecuencia de muestreo nunca se adivina.** Si el archivo no la trae y no pasas `sample_rate_hz`, el lector da
  error. `read_waveform_csv` con columna de tiempo ajusta la frecuencia por mínimos cuadrados y rechaza una columna no
  uniforme.
- **Unidades.** Por dentro todo es SI (mm/s, Pa, s). Las unidades se infieren de la resolución solo para los escalones
  del Micromate y el Minimate Pro (0,00788 mm/s, 0,0156 Pa), con un aviso; en otro caso pasa `units=`.
- **Canales renombrados:** los lectores de Instantel no los reconocen; usa
  `read_waveform_csv(..., columns=..., units=..., sep="\t")` (ahí `units=` es obligatorio).
- **Frecuencia por cruce por cero.** `analyze(zc_method="interpolated")` (por omisión) interpola los cruces;
  `"instantel"` cuenta muestras enteras y marca una frecuencia sobre 100 fs / 1024 Hz. La frecuencia es NaN cuando el
  borde del registro corta el medio ciclo del pico o, con `"instantel"`, sobre ese tope. El veredicto es entonces
  `not_comparable` contra una curva que depende de la frecuencia, nunca una suposición.
- **Suma vectorial.** `pvs_mm_s` es la suma vectorial real (máximo de la suma de cuadrados muestra a muestra); la seudo
  suma vectorial es la raíz de la suma de los cuadrados de los tres picos de componente, una cota superior de la real.
- **FFT.** `fft_frequency` toma la frecuencia dominante de todo el registro y lista los picos a 6 dB del mayor.
- **Aceleración y desplazamiento.** Dependen de la precisión del archivo exportado: con tres cifras significativas o
  menos el resumen avisa. El desplazamiento integra cada medio ciclo como la suma de V por el paso de tiempo y divide
  entre dos el mayor, como describe el manual de Instantel.

## Airblast

`blastingpy.airblast` ajusta la ley del sitio a la sobrepresión pico, la compara con los límites de airblast y trae las
ecuaciones publicadas. Los hechos, las páginas y las marcas están en
[docs/investigacion/FUENTES_SPRINT3.md](docs/investigacion/FUENTES_SPRINT3.md).

```python
from blastingpy import airblast, waveform

events = waveform.events_from_records(records, distance_m=[150, 200, 250], charge_kg=[40, 80, 60])
law = airblast.fit(events)                       # ley de raíz cúbica sobre airblast_pa
limit = airblast.limit_db("OSMRE-30CFR816.67", system="2Hz")
charge = airblast.max_charge(law, 400.0, limit.pressure_pa)
verdict = airblast.evaluate(events, "OSMRE-30CFR816.67", system="2Hz")
```

Una versión completa y ejecutable con datos sintéticos está en [examples/03-airblast](examples/03-airblast/).

- **Unidades.** Por dentro todo es Pa. `airblast.to_db` / `from_db` convierten con P0 = 20 µPa; `read_events(...,
  units={"pressure": "Pa" | "kPa" | "psi" | "dB"})` y `units.to_si` aceptan Pa, kPa y psi. Los eventos traen
  `airblast_pa` y `airblast_freq_hz`; `waveform.events_from_records` los llena desde el micrófono.
- **Ley del sitio.** `airblast.fit` es la regresión de la ley de vibraciones con la forma de raíz cúbica,
  P = K (D / W^(1/3))^-beta, sobre `airblast_pa`: la misma estadística, con cota superior K 10^(z sigma) (la Tabla B-1
  de RI 8485 ajusta el airblast así). `predict`, `max_charge` y `distance_for` funcionan como en `attenuation`.
  `outliers` y `residuals` se refieren a las filas usadas en el ajuste.
- **Una campaña solo de airblast** se pasa como un DataFrame con `distance_m`, `charge_kg` y `airblast_pa`.
- **El sistema de medición es obligatorio.** La librería se niega a comparar con un límite sin él.
- **Un nivel ponderado C lento no es un pico lineal:** con `system="c-slow"` todo veredicto es `not_comparable`, y también una ecuación publicada C-lento frente a cualquier límite. Una ecuación publicada es una mediana: `predict_published` y `evaluate_prediction` avisan siempre que cerca de la mitad de las voladuras la superan.

| Sistema | USBM RI 8485 (1980) | OSMRE 30 CFR 816.67 |
|---|---|---|
| `0.1Hz` (corte inferior de 0,1 Hz o menos) | 134 dB | 134 dB, solo si la autoridad reguladora lo aprueba |
| `2Hz` | 133 dB | 133 dB |
| `6Hz` (un sistema de 5 Hz usa esta fila) | 129 dB | 129 dB |
| `c-slow` (ponderado C, lento) | 105 dB, eventos de hasta 2 s | 105 dB, solo si la autoridad reguladora lo aprueba |

`airblast.system_from_cutoff(low_cutoff_hz)` da el sistema de un micrófono lineal. El límite depende del corte
inferior porque un nivel en dB es el pico de lo que el micrófono deja pasar: cuanto más bajo el corte, más presión de
baja frecuencia cuenta, y por eso cada sistema tiene su propio límite.

Las ecuaciones publicadas (`list_equations`, `published`, `predict_published`) son medianas, sin cota superior, atadas
al sistema con que se ajustaron. Catorce vienen de la Tabla B-1 de RI 8485, dos del BLEP de OSMRE y una de AS 2187.2.

| Id de la ecuación | Sistema | Marca |
|---|---|---|
| `USBM-RI8485/coal-highwall/{0.1Hz,2Hz,5Hz,c-slow}` | el del nombre | V1 |
| `USBM-RI8485/coal-parting/{0.1Hz,2Hz,5Hz,c-slow}` | el del nombre | V1 |
| `USBM-RI8485/metal-mine-highwall/0.1Hz` | 0,1 Hz | V1 |
| `USBM-RI8485/quarry/0.1Hz`, `USBM-RI8485/quarry-direction-of-initiation/0.1Hz`, `USBM-RI8485/quarry-behind-face/0.1Hz`, `USBM-RI8485/quarry-opposite-initiation/0.1Hz`, `USBM-RI8485/quarry-front-of-face/0.1Hz` | 0,1 Hz | V1 |
| `OSMRE-BLEP/highwall`, `OSMRE-BLEP/parting` | no se dice | V1 |
| `AS2187.2-2006/free-face` | no se dice | V2 |

Las dos ecuaciones del BLEP salen de ajustes de 0,1 Hz (lo dice RI 8485; el BLEP mismo no dice el sistema), pero el BLEP las compara con el límite de 2 Hz;
`evaluate_prediction` avisa cuando el sistema de la ecuación difiere del del límite. AS 2187.2 es [V2], de una sola
fuente secundaria.

No entran: la corrección por viento, los límites de confort australianos y AS 2187.2 desde la norma misma.

## Flyrock

`blastingpy.flyrock` trae los modelos publicados de proyección de rocas, una tabla comparativa por pozo y la distancia de
despeje. Los hechos, las páginas y las marcas están en
[docs/investigacion/FUENTES_SPRINT4.md](docs/investigacion/FUENTES_SPRINT4.md).

```python
import pandas as pd
from blastingpy import flyrock

holes = pd.DataFrame({"diameter_mm": [165.0, 165.0], "burden_m": [4.0, 3.2], "stemming_m": [3.0, 2.5],
                      "linear_charge_kg_m": [18.2, 18.2], "hole_angle_from_vertical_deg": [10.0, 10.0]})
table = flyrock.compare(holes, k=27.0)               # alcances en m, una fila por pozo
worst = table["rm_max_m"].max()
clear = flyrock.clearance(worst, target="people", regulation="PE-DS024-2016-EM")
print(clear.distance_m, clear.governs)
```

Una versión completa y ejecutable, con una tabla de pozos sintética, está en [examples/04-flyrock](examples/04-flyrock/).

- **Ningún modelo trae margen de seguridad; `clearance` lo agrega:** factor 2 para equipos y 4 para personas (Richards y Moore), o un factor propio >= 1.
- **Mínimo peruano:** DS 024-2016-EM art. 267 a), modificado por el DS 034-2023-EM: radio de 500 m cuando la roca puede proyectarse 360°, y en voladuras de banco al menos 500 m en la dirección en que la voladura lanza. Rige para la evacuación de los trabajadores (personas, no equipos); el resultado dice cuál manda, el modelo o el reglamento (`governs`). Reportar el mayor entre el alcance con factor y 500 m es una decisión de diseño, no el texto del reglamento.
- **`k` es obligatorio** (constante del sitio de Richards y Moore): 13,5 para roca competente blanda, 27 para roca competente dura; el rango de los datos publicados es 15 a 37.
- **Columna de carga menor de 1 m:** `compare` usa la masa real de la carga (kg) en vez de la carga lineal, como prescriben Richards y Moore, y avisa.
- **El ángulo del taladro** se mide desde la vertical. Un taladro vertical da 0 m de eyección de taco, con un aviso (la fuente dice que los verticales necesitan un ajuste).
- **Los diámetros van en mm** (la fuente de Lundborg usa pulgadas; la librería convierte y rechaza valores que parecen pulgadas o metros).
- Los modelos son estimaciones distintas, no un consenso: nada los promedia ni elige uno.

| Modelo | Qué estima | Entradas | Marca | Fuente leída |
|---|---|---|---|---|
| Balística, `FLY-BALLISTIC` (`range_level`, `max_range_level`, `range_to_height`, `max_height`, `velocity_for_range`) | Alcance y altura sin arrastre | Velocidad, ángulo de lanzamiento | V1 | Roth 1979 ecs. (1)-(3), (5); Richards y Moore ecs. [4]-[5] |
| Richards y Moore, `FLY-RM-FACE`, `FLY-RM-CRATER`, `FLY-RM-STEMMING` (`rm_face_burst`, `rm_cratering`, `rm_stemming_ejection`, `rm_min_confinement`) | Alcance máximo por frente, cráter y eyección de taco; burden o taco mínimo | `k`, carga (kg/m), burden, taco, ángulo | V1 | Richards y Moore, ISEE, ecs. [6], [7a]-[7c] |
| Lundborg, `FLY-LUNDBORG-MAX`, `FLY-LUNDBORG-Q` (`lundborg_max_range`, `lundborg_optimal_fragment`, `lundborg_range`) | Alcance máximo y fragmento óptimo; alcance por factor de carga | Diámetro (mm), factor de carga | V2 | Reexpresado por Richards y Moore, McKenzie 2018 y Szendrei y Tose 2022; originales no leídos |
| McKenzie 2018, `FLY-MCKENZIE-2018` (`mckenzie_max_range`) | Alcance máximo | SDoB (lo das tú), diámetro | V1 | Resumen de McKenzie 2018, lámina 11, solo la versión 2018 |
| Roth, `FLY-ROTH-GURNEY`, `FLY-ROTH-CAL` (`roth_charge_mass_ratio`, `roth_gurney_velocity`, `roth_launch_velocity`, `roth_calibrated_velocity`, `roth_reference_gurney_velocity`) | Velocidad de lanzamiento desde c/m y la velocidad de Gurney; ecuaciones calibradas | Carga lineal, densidad de la roca, burden, velocidad de detonación | V1 | Roth 1979, USBM OFR 77-81, ecs. (7)-(10), (13), (13a), (14), (17) |

`flyrock.info(flyrock.Model.X)` da el id, la marca y la cita de cada uno, y `compare` los lleva en `attrs`.

**Críticas publicadas** (Szendrei y Tose 2022, JSAIMM 122(12), pp. 726-731, `attrs["critiques"]`): los modelos sin arrastre pueden sobrestimar el alcance en factores de 2 a 8; Lundborg vale para fragmentos esféricos del tamaño óptimo; McKenzie hereda la calibración de Lundborg.

**No entra, y por qué:** el coeficiente de McKenzie (2009) y la definición de SDoB (ninguna fuente leída la da, así que el SDoB lo pones tú), el tamaño de fragmento y el taco mínimo de McKenzie, las curvas gráficas y la correlación de banco de Roth, escalar las ecuaciones calibradas de Roth a otros explosivos (la fuente no dice la regla y su factor impreso parece invertido), las trayectorias con arrastre, la forma de la zona de despeje y un lector de la tabla de pozos desde archivo (se pasa un DataFrame; pandas ya lee CSV y Excel).

## Convenciones

- **Unidades.** Dentro de la librería todo es SI: m, kg, mm/s, Hz. `read_events(units="imperial")` (o un diccionario por
  magnitud) convierte al leer; `blastingpy.units` tiene `to_si` y `from_si`.
- **Cota superior de la ley del sitio.** `k_upper = k * 10^(z * sigma)` con `z = 2` por omisión, como en USBM RI 8507 y
  en la práctica de BIVDEP de OSMRE. Para la cota unilateral del 95 % usa
  `attenuation.fit(events, z=attenuation.normal_quantile(0.95))` (z = 1,645). `max_charge` y `predict_ppv` usan la cota
  superior por omisión; pasa `attenuation.Level.Median` para la mediana.
- **Avisos.** Un ajuste de vibración con menos de 30 eventos (el de sobrepresión no avisa por eso), con `beta <= 0`, o una predicción fuera del rango de distancia o de
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
