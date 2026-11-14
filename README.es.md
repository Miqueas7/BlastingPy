# BlastingPy

Librería abierta de Python, con núcleo en C++20, para analizar los efectos de la voladura en minería: vibración del
terreno, sobrepresión de aire, proyección de rocas y vibración en campo cercano.

El caso de uso central: al terminar una campaña de monitoreo, el ingeniero de voladura tiene una tabla de eventos
(distancia, carga máxima por retardo, PPV, frecuencia). BlastingPy ajusta la ley de atenuación del sitio, calcula la
carga máxima por retardo y verifica el resultado contra un criterio normativo, indicando la edición, la tabla, la
magnitud y el método de frecuencia que usó.

**Estado:** en desarrollo, sin publicar (versión 0.1.0). Los cinco bloques de la versión 0.1 están completos: tabla de
eventos, ley de atenuación, carga máxima y veredicto de cumplimiento; lectura de registros del sismógrafo (forma de onda,
PPV, suma vectorial, frecuencia); sobrepresión de aire; proyección de rocas (modelos publicados de alcance, comparación
por pozo y distancia de despeje); y vibración en campo cercano (Holmberg-Persson, Heelan, simulación de la voladura
completa y su Monte Carlo). El paquete no está en PyPI y el repositorio es privado.

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
  uniforme; una frecuencia ajustada a menos de 1e-6 de un entero, y un predisparo a menos de 1e-3 de un periodo de
  muestreo de una muestra entera, se toman como esos valores (tiempos redondeados al microsegundo dan
  1024,0000003 Hz). SAC guarda el paso de muestreo redondeado, así que un SAC de 1024 Hz se lee como 1023,54 Hz:
  `read_obspy` se queda con la frecuencia del archivo, agrega un aviso que nombra el entero y acepta `sample_rate_hz=`
  (debe coincidir con el archivo dentro del 0,1 %).
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
  Cuando el pico dominante cae en los dos bins más bajos (f <= 2 / T, T la duración del registro), como con varios
  pozos que interfieren, el espectro lo domina la envolvente del registro o una tendencia, no una oscilación:
  `fft_frequency` avisa (`UserWarning`) y `analyze` lo agrega a sus avisos. Usa entonces la frecuencia por cruces por
  cero.
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
- **Veredictos como en `criteria.evaluate`:** `airblast.evaluate` da `not_comparable` a un evento sin valor de airblast (`airblast_pa` vacío o NaN) e `indeterminate` a un valor fuera del dominio (<= 0 o infinito).
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
worst = table["max_model_m"].max()                   # el peor pozo, mayor alcance propio del pozo
clear = flyrock.clearance(worst, target="people", regulation="PE-DS024-2016-EM")
print(clear.distance_m, clear.governs)
```

Una versión completa y ejecutable, con una tabla de pozos sintética, está en [examples/04-flyrock](examples/04-flyrock/).

- **Ningún modelo trae margen de seguridad; `clearance` lo agrega:** factor 2 para equipos y 4 para personas (Richards y Moore), o un factor propio >= 1.
- **Mínimo peruano:** DS 024-2016-EM art. 267 a), modificado por el DS 034-2023-EM: radio de 500 m cuando la roca puede proyectarse 360°, y en voladuras de banco al menos 500 m en la dirección en que la voladura lanza. Rige para la evacuación de los trabajadores (personas, no equipos); el resultado dice cuál manda, el modelo o el reglamento (`governs`). Reportar el mayor entre el alcance con factor y 500 m es una decisión de diseño, no el texto del reglamento.
- **`k` es obligatorio** (constante del sitio de Richards y Moore): 13,5 para roca competente blanda, 27 para roca competente dura; el rango de los datos publicados es 15 a 37. Un `k` fuera de [13,5, 37] da un aviso (el alcance va con k²).
- **Tabla de pozos:** columnas obligatorias `diameter_mm`, `burden_m`, `stemming_m`, `linear_charge_kg_m` (o `explosive_density_kg_m3`, 300-2000 kg/m³, para calcularla con el pozo lleno); opcionales `charge_length_m`, `hole_angle_from_vertical_deg`, `powder_factor_kg_m3` (alcance de Lundborg por factor de carga) y `sdob_m_kg13` (McKenzie, en m/kg^(1/3)). Un NaN en una columna opcional quiere decir «no se dio para esa fila».
- **`max_model_m`** es el mayor alcance entre los modelos propios de cada pozo (`rm_face_burst_m`, `rm_cratering_m`, `rm_stemming_ejection_m`, `lundborg_q_m`, `mckenzie_2018_m`); `compare` avisa de los pozos donde pasa del doble de `rm_max_m`. **`lundborg_max_m`** queda aparte: depende solo del diámetro (el máximo práctico de Lundborg, para el fragmento óptimo), así que es el mismo en todos los pozos de un diámetro y no dice cuál es el peor; es la envolvente superior que da el diámetro, y `compare` avisa de los pozos donde pasa del doble de `max_model_m`. El despeje de la voladura lo manda su peor pozo, y elegir el modelo es decisión del ingeniero: `clearance` recibe un solo alcance y lo dice en sus notas.
- **Avisos donde el resultado puede salir corto:** el alcance de Lundborg por factor de carga menor que su máximo (en `compare`, menor que la mitad, umbral de diseño; una correlación con factor de carga tiende a subestimar, Roth 1979), un SDoB mayor de 2,4 (¿ft/lb^(1/3) en vez de m/kg^(1/3)?), un taladro vertical, las ecuaciones calibradas de Roth cerca de su c/m mínimo y una carga lineal que implica una densidad menor de 300 kg/m³. Una densidad en g/cc es error.
- **Columna de carga menor de 1 m:** `compare` usa la masa real de la carga (kg) en vez de la carga lineal, como prescriben Richards y Moore, y avisa.
- **El ángulo del taladro** se mide desde la vertical. Un taladro vertical da 0 m de eyección de taco, con un aviso (la fuente dice que los verticales necesitan un ajuste).
- **Los diámetros van en mm** (la fuente de Lundborg usa pulgadas; la librería convierte y rechaza valores que parecen pulgadas o metros).
- Los modelos son estimaciones distintas, no un consenso: nada los promedia ni elige uno.

| Modelo | Qué estima | Entradas | Marca | Fuente leída |
|---|---|---|---|---|
| Balística, `FLY-BALLISTIC` (`range_level`, `max_range_level`, `range_to_height`, `max_height`, `velocity_for_range`) | Alcance y altura sin arrastre | Velocidad, ángulo de lanzamiento | V1 | Roth 1979 ecs. (1)-(3), (5); Richards y Moore ecs. [4]-[5]; Moore y Richards 2005 ecs. [7], [9], [10] |
| Richards y Moore, `FLY-RM-FACE`, `FLY-RM-CRATER`, `FLY-RM-STEMMING` (`rm_face_burst`, `rm_cratering`, `rm_stemming_ejection`, `rm_min_confinement`) | Alcance máximo por frente, cráter y eyección de taco; burden o taco mínimo | `k`, carga (kg/m), burden, taco, ángulo | V1 | Richards y Moore, ISEE, ecs. [6], [7a]-[7c] |
| Lundborg, `FLY-LUNDBORG-MAX`, `FLY-LUNDBORG-Q` (`lundborg_max_range`, `lundborg_optimal_fragment`, `lundborg_range`) | Alcance máximo y fragmento óptimo; alcance por factor de carga | Diámetro (mm), factor de carga | V2 | Reexpresado por Richards y Moore, McKenzie 2018 y Szendrei y Tose 2022; originales no leídos |
| McKenzie 2018, `FLY-MCKENZIE-2018` (`mckenzie_max_range`) | Alcance máximo | SDoB (lo das tú), diámetro | V1 | Resumen de McKenzie 2018, lámina 11, solo la versión 2018 |
| Roth, `FLY-ROTH-GURNEY`, `FLY-ROTH-CAL` (`roth_charge_mass_ratio`, `roth_gurney_velocity`, `roth_launch_velocity`, `roth_calibrated_velocity`, `roth_reference_gurney_velocity`) | Velocidad de lanzamiento desde c/m y la velocidad de Gurney; ecuaciones calibradas | Carga lineal, densidad de la roca, burden, velocidad de detonación | V1 | Roth 1979, USBM OFR 77-81, ecs. (7)-(10), (13), (13a), (14), (17) |

`flyrock.info(flyrock.Model.X)` da el id, la marca y la cita de cada uno, y `compare` los lleva en `attrs`.

**Críticas publicadas** (`attrs["critiques"]`): los modelos sin arrastre pueden sobrestimar el alcance en factores de 2 a 8; Lundborg vale para fragmentos esféricos del tamaño óptimo; McKenzie hereda la calibración de Lundborg (Szendrei y Tose 2022, JSAIMM 122(12), pp. 726-731); una correlación con factor de carga tiende a subestimar el alcance (Roth 1979, pp. 10-11, 29-31).

**No entra, y por qué:** el coeficiente de McKenzie (2009) y la definición de SDoB (ninguna fuente leída la da, así que el SDoB lo pones tú), el tamaño de fragmento y el taco mínimo de McKenzie, las curvas gráficas y la correlación de banco de Roth, escalar las ecuaciones calibradas de Roth a otros explosivos (la fuente no dice la regla y su factor impreso parece invertido), las trayectorias con arrastre, la forma de la zona de despeje y un lector de la tabla de pozos desde archivo (se pasa un DataFrame; pandas ya lee CSV y Excel).

## Campo cercano

`blastingpy.nearfield` estima la vibración cerca de una carga (Holmberg-Persson, Heelan) y simula una voladura completa
superponiendo una onda semilla, con un Monte Carlo sobre los retardos y las amplitudes. Los hechos, las páginas y las
marcas están en [docs/investigacion/FUENTES_SPRINT5.md](docs/investigacion/FUENTES_SPRINT5.md).

```python
import pandas as pd
from blastingpy import attenuation, datasets, nearfield

# Holmberg-Persson alrededor de una carga de 3 m y 1 kg/m. K, alfa y beta los pones tú.
charge = dict(k=700.0, alpha=0.7, beta=1.5, linear_charge_kg_m=1.0, z_start=-1.5, z_end=1.5)
critical = nearfield.ppv_critical(10e6, 5000.0, 50e9)       # T Vp / E con el E estático: 1000 mm/s
extent = nearfield.hp_damage_extent(nearfield.HpModel.Niosh, **charge, z=0.0, ppv_limit_mm_s=critical)

# Heelan alrededor de una columna de 6 m cebada en el fondo.
source = nearfield.PressureSource(vn_pressure_pa=5e9, n=6, gamma_per_s=8000.0)
medium = nearfield.Medium(p_velocity_m_s=5000.0, s_velocity_m_s=2900.0, density_kg_m3=2700.0)
column = nearfield.Column(hole_radius_m=0.05, z_bottom=0.0, z_top=6.0, elements=41, primer_element=1,
                          vod_m_s=4500.0, charge_exponent=1.0)
peak = nearfield.heelan_ppv(source, medium, column, 5.0, 3.0)        # en r = 5 m, z = 3 m
print(peak.ppv_mm_s, peak.warnings)

# Una voladura completa: tabla de pozos, semilla (el registro de un disparo de un solo pozo) y ley del sitio.
holes = pd.DataFrame({"x_m": [0.0, 5.0, 10.0], "y_m": 0.0, "charge_kg": 80.0, "firing_time_s": [0.0, 0.017, 0.034]})
seed = (datasets.synthetic_record(pretrigger_s=0.0, mic_pa=None), "vertical")   # en lugar de tu registro
blast = dict(law=attenuation.fit(datasets.synthetic_events(), law="sqrt"), wave_velocity_m_s=3000.0)
result = nearfield.simulate_blast(holes, (5.0, 120.0), seed, **blast)
print(result.ppv_mm_s, result.arrivals, result.warnings)
mc = nearfield.monte_carlo_blast(holes, (5.0, 120.0), seed, delay_sigma_s=0.002, amplitude_cv=0.25,
                                 simulations=2000, rng_seed=1, **blast)
print(mc.percentile(95), mc.probability_exceeding(25.0))
```

Una versión completa y ejecutable con datos sintéticos está en [examples/05-nearfield](examples/05-nearfield/).

| Modelo | Qué estima | Entradas | Marca | Fuente leída |
|---|---|---|---|---|
| Holmberg-Persson, forma integral original (`hp_point`, `hp_original`, `hp_grid("original", ...)`) | PPV estático alrededor de una carga larga, K [q ∫ dz / (r² + z²)^(β/(2α))]^α | K, α, β (los das tú), carga lineal (kg/m), tramo cargado y punto (m) | V1, a través de NIOSH | Iverson, Kerkering y Hustrulid (2008), NIOSH, ecs. 1 y 5. Los artículos de Holmberg y Persson no se leyeron |
| Holmberg-Persson, forma cerrada con β = 2α (`hp_term`) | La misma integral como arcotangente; `hp_original` la usa cuando β = 2α | Carga lineal, tramo cargado, punto | V2 | Onederra y Esen (2004), Fragblast 8(2), ec. 2 |
| Forma modificada de NIOSH (`hp_mean_distance`, `hp_niosh`, `hp_grid("niosh", ...)`) | K (q L)^α / R̄^β, con R̄ la distancia media de la carga al punto | Las mismas de la forma original | V1 | NIOSH (2008), ecs. 8-10 |
| PPV crítico (`ppv_critical`) y extensión del daño (`hp_damage_extent`) | T Vp / E, y la distancia radial a la que cualquiera de las dos formas baja a un PPV límite | Resistencia a tracción y módulo de Young estático (Pa), velocidad P (m/s); forma, constantes, carga, altura z, límite (mm/s) | V2; la extensión lleva la marca de la forma que se use | Onederra y Esen (2004), ec. 11, que citan a Persson, Holmberg y Lee (1994, p. 247). Que el módulo sea el estático es inferido |
| Elemento de Heelan (`heelan_element`, con `pressure_function`, `g_function`, `g_peak`, `scaled_frequency`, `heelan_amplitude`) | Onda de velocidad P y SV que radia una carga cilíndrica corta | `PressureSource` (presión de von Neumann en Pa, n, decaimiento γ en 1/s), `Medium` (Vp y Vs en m/s, densidad en kg/m³), radio del taladro, media longitud, punto, instantes | V1 | Heelan (1953), Geophysics 18(3), ecs. 12-13, en la forma de Blair (2007), Geophysics 72(2), ecs. 8 y 13-16 |
| Columna de Heelan (`Column`, `column_weights`, `heelan_column`, `vector_peak`, `heelan_ppv`) | Onda y PPV vectorial de una columna que inicia a la velocidad de detonación, lineal (A = 1) o con pesos no lineales (A < 1) | Fuente, medio, `Column` (radio del taladro, tramo, número impar de elementos, elemento del cebo, velocidad de detonación en m/s, exponente de carga A), punto | V1 | Blair (2008), IJRMMS 45, ecs. 1-3 y 22. Inferido: el instante de iniciación de cada elemento y el reflejo de los pesos cuando el cebo está sobre la mitad de la columna |
| Superposición de la onda semilla (`superpose`, `simulate_blast`, `delay_sweep`) | Onda de una voladura completa en un monitor: una semilla por pozo, retardada y escalada | Tabla de pozos, monitor, semilla, ley del sitio, velocidad de onda | V1 | Hinzen (1988), IJRMMS 25(6), ecs. 3-7; Blair (1999), Fragblast 3(4), ec. 1a |
| Pesos de voladura completa (`screening_counts`, `screening_factor`, `charge_scaling_weights`, `damage_weights`) | Peso de cada pozo por apantallamiento, escala de carga o daño | Posiciones, cargas, tiempos de iniciación, espaciamiento medio (m) | V1 | Blair (1999), ecs. 10b-10c; Blair (2008), ecs. 7-12 y 16-21 |
| Monte Carlo (`monte_carlo_peaks`, `monte_carlo_blast`) | Distribución del pico con retardos y amplitudes dispersos | Las mismas de la simulación, más `delay_sigma_s` (s), `amplitude_cv`, `simulations`, `rng_seed` | V1 para las ecuaciones; la distribución de retardos la pones tú | Blair (1999), ecs. 1b y 7 |
| Carga por ventana (`max_charge_per_window`, `max_charge_8ms_kg`) | Mayor carga que inicia dentro de 8 ms | Tiempos de iniciación (s), cargas (kg) | V2 | Duvall et al. (1963), según Blair (1999, p. 337) |

- **Unidades.** Longitudes, coordenadas y espaciamiento en m; cargas en kg y carga lineal en kg/m; tiempos de iniciación, retardos y su dispersión en **segundos, no en milisegundos**; velocidades de onda y de detonación en m/s; presiones y módulos en Pa; densidad en kg/m³; frecuencia de muestreo en Hz. Todo PPV y toda onda salen en mm/s (las funciones `hp_*` devuelven la unidad que lleve K: mm/s con constantes ajustadas en mm/s, kg y m).
- **K, α y β de Holmberg-Persson son obligatorios.** Ninguna función tiene un valor por omisión para ellos. 700 / 0,7 / 1,5 son los valores que NIOSH reporta para voladuras en superficie ([V2], atribuidos a Lundborg et al. 1978; no se nombra un tipo de roca); se ofrecen como dato en `HOLMBERG_PERSSON_SURFACE`, con su cita, y solo se usan si los pasas. Ajusta las constantes de tu sitio.
- **Holmberg-Persson es estático y su integral tiene un defecto conocido.** No depende del tiempo e ignora la velocidad de detonación (Onederra y Esen 2004, p. 62, que resumen a Blair y Minchinton 1996 y 1997). Al pasar de la suma sobre los elementos de carga a la integral, el exponente α sale de dentro de la suma a fuera de la integral, y el exponente no conmuta con la suma (Hustrulid y Lu 2002, según NIOSH, PDF p. 7; Blair 2008, p. 235, señala el mismo defecto). La forma modificada de NIOSH lo resuelve con una distancia media. La forma original diverge sobre el eje cargado (`hp_original` da error ahí y `hp_grid` da NaN); la modificada es finita en todas partes.
- **Heelan vale dentro de ciertos límites, y la librería avisa fuera de ellos.** Blair (2007, pp. E37-E40): la frecuencia escalada Ω_A = c γ a / Vp debe ser a lo más de alrededor de 0,1 (por encima, Heelan sobrestima la solución exacta a cualquier distancia); el punto está en campo cercano cuando Ω_A r / a < π; y la media longitud de un elemento no debe pasar de unas dos veces el radio. `a` es el radio del taladro, la convención de Blair (2007); Heelan (1953, p. 695) lo define como el radio de la cavidad equivalente. `heelan_ppv` no rechaza nada por eso: cada límite que no se cumple va en `warnings` con el número calculado, junto a las advertencias fijas (P y SV se suman linealmente donde se solapan; los pesos no lineales de la columna no tienen validación experimental).
- **Grilla de tiempo de `heelan_ppv`.** El PPV es siempre el pico de la onda. Sin `sample_rate_hz` ni `duration_s` la grilla es automática: un paso de 1 / (40 γ) hasta la última llegada más 40 / γ (o más (n + 4√n + 2) / γ si es mayor). Con tu propia grilla avisa cuando el registro termina antes de la última llegada más (n + 4√n + 2) / γ, lo que dura el pulso (el pico puede salir truncado) y cuando el paso es mayor de 1 / (10 γ) (el pulso queda mal resuelto; la frecuencia de muestreo de un sismógrafo suele ser demasiado baja para esto).
- **Simulación de la voladura completa.** La amplitud de cada pozo es la predicción **mediana** de la ley del sitio (no su cota superior) para su carga y su distancia al monitor, por un peso; cada pozo llega en su tiempo de iniciación más distancia / velocidad de onda; el resultado es una sola componente, la de la semilla. La semilla se normaliza a pico unitario y se usa desde su primera muestra: recorta el predisparo. Revisa `arrivals` antes de confiar en el pico. Una llegada entre dos muestras se reparte linealmente entre ellas, lo que baja su pico hasta en sin²(π f / fs) para una semilla de frecuencia dominante f (9,5 % con f = fs / 10): muestrea la semilla a diez veces su frecuencia dominante o más.
- **Tres ponderaciones, cada una con la advertencia de su autor** (viajan en `warnings`). `"linear"` (Hinzen 1988; Blair 1999), con el apantallamiento opcional de Blair (1999), cuyo g se ajustó a un solo conjunto de datos de campo y cuyo piso de 0,2 pone el autor por falta de datos de campo cercano. `"charge-scaling"` (Blair 2008, ec. 12): su autor reporta una caída casi lineal e irreal de la amplitud con el tiempo y no la recomienda frente al modelo de daño. `"damage"` (Blair 2008, ec. 21): η = 0,2 es la primera estimación del autor y 0,3 ajustó su único caso de campo. Los modelos de Blair (2008) no tienen caras libres y no cambian la forma de la onda. `charge_exponent` toma por omisión el de la ley (β/2 con ley de raíz cuadrada, β/3 con raíz cúbica, el exponente ajustado con la ley general); `SiteLaw.charge_exponent` no es ese número en las leyes de distancia escalada.
- **Los tiempos de iniciación se comparan con una tolerancia de 1 ns.** Los pozos a menos de 1 ns entre sí son simultáneos: no cuentan como ya iniciados uno para el otro, y dos pozos nominalmente a 8 ms nunca comparten ventana por redondeo.
- **La ventana de 8 ms.** `max_charge_8ms_kg` es la mayor carga que inicia dentro de 8 ms cualesquiera, por tiempos de iniciación. La ventana viene de Duvall et al. (1963), según Blair (1999, p. 337), quien agrega que no hay nada especial en 8 ms: que dos pozos se separen depende de la semilla.
- **Qué aleatoriza el Monte Carlo.** La llegada de cada pozo, con una dispersión normal de desviación estándar `delay_sigma_s` que das tú (decisión de diseño: Blair 1999 no publica ni el tipo ni la desviación de la distribución de retardos), y su amplitud, multiplicada por (1 + c z) con z normal estándar y c = `amplitude_cv` (Blair 1999, ec. 7). No aleatoriza la forma de la semilla (la semilla fluctuante de Blair), así que la variabilidad de la onda queda subrepresentada, y los pesos salen de los tiempos de iniciación nominales. La misma `rng_seed` da los mismos picos (entre plataformas, a cerca de 1e-9 relativo). El valor por omisión `amplitude_cv = 0` ignora la dispersión de la ley del sitio: si la ley tiene σ > 0, un aviso da el coeficiente de variación de una lognormal con ese σ, √(exp((σ ln 10)²) − 1), para que lo pases.
- **Avisos de plausibilidad.** Tiempos de iniciación que abarcan más de 20 s o con más de 1 s entre pozos consecutivos, y un retardo mayor de 1 s en `delay_sweep`, parecen milisegundos y se señalan; también una `wave_velocity_m_s` menor de 100 (parece estar en km/s). Un `delay_sigma_s` mayor de 1 s es error, igual que una onda de más de 2 000 000 de muestras. También se señalan una semilla tomada del canal del micrófono, una semilla cuya frecuencia dominante pasa de la décima parte de su frecuencia de muestreo y un argumento que la ponderación elegida no usa.
- **Avisos de lo que un PPV más bajo escondería.** `simulate_blast` y `monte_carlo_blast` dicen siempre que las amplitudes son la mediana de la ley del sitio (cerca de la mitad de los pozos la superan). Con `weighting="damage"` se listan los pozos con un factor de apantallamiento S_n menor de 0,2 o a menos de 99 m del monitor: S_n (Blair 2008, ec. 20) no tiene piso y g se ajustó con monitores a 99 m o más (Blair 1999, Fig. 7b), así que sus pesos extrapolan la fuente. Estos avisos (unidades, la mediana, `amplitude_cv`, el campo cercano del modelo de daño y la semilla submuestreada) van en `warnings` y además se emiten como `UserWarning`, para que se vean aunque nunca leas `warnings`.
- **Sin validación contra datos de campo.** Las funciones reproducen las ecuaciones y los ejemplos numéricos de sus fuentes; ninguna simulación de aquí se ha comparado con una voladura medida.

**No entra, y qué artículo lo destrabaría:** el modelo de Heelan escalado y su constante γ_n (Blair y Minchinton 2006, Fragblast-8, pp. 152-159; la columna de aquí es el elemento de Heelan estándar con los pesos de Blair 2008); la solución exacta de la carga corta (Blair 2007 y Meredith 1990; leída, fuera del alcance); la semilla fluctuante del Monte Carlo (su normalización se deriva en Blair 1993a, Int. J. Numer. Anal. Methods Geomech. 17: 95-118); la composición de las componentes longitudinal, transversal y vertical (Blair 1999, ecs. 15a-15c: la convención de signos depende de una figura); la carga estadística y su espectro (Blair 1999, ecs. 17-23: leídas, sin implementar; la ec. 23 impresa trae un factor que su propio límite no da); la partición de energía de Heelan y su razón SV/P de 1,6 (las fórmulas impresas son inconsistentes); el PPV al final de la zona triturada (Onederra y Esen 2004, ec. 5, con Liu y Katsabanis 1993 y Esen et al. 2003: un exponente no se pudo verificar y pide presiones de detonación no ideal); y los artículos de Holmberg y Persson (1979) y Persson, Holmberg y Lee (1994, p. 247), con los que la integral, las constantes de superficie y el PPV crítico pasarían a [V1]. Su límite de daño de 700-1000 mm/s es de rocas ígneas suecas, según NIOSH, y no se ofrece como criterio general.

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

Los cinco bloques previstos para la versión 0.1 ya están; lo que sigue no forma parte de ella. Cada bloque lista lo que
deja fuera, y por qué, al final de su sección ([Airblast](#airblast), [Flyrock](#flyrock),
[Campo cercano](#campo-cercano)). En vibración del terreno: las leyes Langefors-Kihlström, Ghosh-Daemen y CMRI, y las
normas UNE 22381, AS 2187.2, BS 6472-2, SS 4604866, SN 640312a y NBR 9653 no entran porque no se leyó ninguna fuente con
sus fórmulas o valores. Tampoco la línea 1 de DIN 4150-3 ni los criterios del último piso, la curva de la Figura 1 de
OSMRE ni la línea 1 de BS 7385-2 bajo 4 Hz. Tres formatos de sismógrafo (THOR, Blastware con encabezado e histograma) no
están validados contra un archivo real. Detalle en [docs/CHANGELOG.md](docs/CHANGELOG.md).

## Licencia

MIT. Ver [LICENSE](LICENSE).
