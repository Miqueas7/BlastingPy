# Flyrock

Hole table -> per-hole ranges of the published models -> governing hole -> clearance distance -> Roth for one hole.

```
python flyrock.py
```

The hole table is synthetic (six invented holes of 165 mm, one with a charge column shorter than 1 m, one vertical); it
says nothing about any real blast. `k = 27` is the Richards & Moore value for hard competent rock; you supply your own.
Nothing is written to disk. The output shows the table of ranges, the warnings, the governing hole, the clearance for
people and equipment with and without the Peruvian regulatory minimum (a mild hole shows the regulation governing),
the minimum stemming for a target range, Roth's launch velocity for one hole and the published critiques.
