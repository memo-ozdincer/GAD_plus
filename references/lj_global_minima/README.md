# Lennard-Jones reference structures

These Cartesian point files were downloaded on 2026-08-09 from the Cambridge
Energy Landscape Database table of reduced-unit Lennard-Jones global minima:

`https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html`

The `global` files are the reported lowest minima. The LJ38 and LJ75
`icosahedral` files are the reported lowest icosahedral competitors from the
same table. Coordinates are in units of `sigma`; the local predictor uses
`epsilon=sigma=1`.

SHA-256 checksums of the downloaded numeric files:

```text
6397b7e9c06f5da94dbfcefe80d122ba068097517ef14454fdf2f2f31699d416  lj13_global.points
45549b4bd8fa82b4a61f9a2ad6343e934448b3abb28f7f897b28a60d908e622f  lj31_global.points
61be40be128a801f1f2e8b8c7c047c87e66676d65831957424a132e95ab58454  lj38_global.points
4a299690068a3f6296074a855c7324299c877fa70cb7e3eb88c1d57d5ffb0a9a  lj38_icosahedral.points
43f5c2568c7de4fc03273427d4c2772a9104e727f22a7dfb60c87d2d333be7fa  lj55_global.points
dcacf738fc99d1bd81cc3de45c6bc3f208af094cb2a035b12cb0e880fe47dd10  lj75_global.points
8f1823a91e2ee7b8aeb5792c95d831318ac747279290209ce8ec0652d727617c  lj75_icosahedral.points
```

The structures are inputs, not optimizer outcomes. Every benchmark must
independently recompute energy, force, and projected index before use.
