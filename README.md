# BlastingPy

Open-source Python library, with a C++20 core, to analyze the effects of blasting in mining: ground vibration,
airblast, flyrock and near-field vibration.

The central use case: a blasting engineer ends a monitoring campaign with a table of events (distance, maximum charge per
delay, PPV, frequency). BlastingPy fits the attenuation law of the site, computes the maximum charge per delay and
checks the result against a regulatory criterion, stating the edition, table, magnitude and frequency method used.

**Status:** in development, unpublished (version 0.1.0, sprint 1 of 5: attenuation law and criteria).
The package is not on PyPI and the repository is private.

## Development install

```bash
python -m pip install ".[test,io,viz]"
python -m pytest tests/python -q
```

A C++ compiler and CMake are required to build from source.

## License

MIT. See [LICENSE](LICENSE).
