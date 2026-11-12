# BlastingPy

Librería abierta de Python, con núcleo en C++20, para analizar los efectos de la voladura en minería: vibración del
terreno, sobrepresión de aire, proyección de rocas y vibración en campo cercano.

El caso de uso central: al terminar una campaña de monitoreo, el ingeniero de voladura tiene una tabla de eventos
(distancia, carga máxima por retardo, PPV, frecuencia). BlastingPy ajusta la ley de atenuación del sitio, calcula la
carga máxima por retardo y verifica el resultado contra un criterio normativo, indicando la edición, la tabla, la
magnitud y el método de frecuencia que usó.

**Estado:** en desarrollo, sin publicar (versión 0.1.0, sprint 1 de 5: ley de atenuación y criterios).
El paquete no está en PyPI y el repositorio es privado.

## Instalación para desarrollo

```bash
python -m pip install ".[test,io,viz]"
python -m pytest tests/python -q
```

Para compilar desde el código fuente hacen falta un compilador de C++ y CMake.

## Licencia

MIT. Ver [LICENSE](LICENSE).
