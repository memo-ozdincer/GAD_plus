# Hybrid GAD-Newton sweep — pivot tables


## 10 pm noise — convergence %  (rows = method, columns = trust radius Å)

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_swfalse                 89.2   88.9   88.9   88.9   88.9
hybrid_eckart_swfalse          45.6   46.3   46.3   46.3   46.3
hybrid_eckart_swtrue           84.7   84.7   84.7   84.7   85.4
hybrid_damped_eckart_swfalse   45.6   46.3   46.3   46.3   46.3
hybrid_damped_eckart_swtrue    86.8   86.8   86.1   85.7   85.4
```

*GAD baselines (5k step budget, no Newton phase) for context:*

- **GAD dt=0.003 (5k)** at 10pm: conv = 89.2%, median step at conv = 164, wall/conv = 54.2 s
- **GAD dt=0.005 (5k)** at 10pm: conv = 89.2%, median step at conv = 98, wall/conv = 47.7 s
- **GAD dt=0.007 (5k)** at 10pm: conv = 89.2%, median step at conv = 72, wall/conv = 46.6 s

### Median steps to converge — 10pm  (lower is better)

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_swfalse                  106    104    104    104    104
hybrid_eckart_swfalse           702    702    702    702    702
hybrid_eckart_swtrue             21     12      8      5      5
hybrid_damped_eckart_swfalse    702    702    702    702    702
hybrid_damped_eckart_swtrue      35     19     10      6      5
```

### Wall-time per converged TS — 10pm  (sec, lower is better)

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_swfalse                 16.1   15.7   15.8   15.9   15.9
hybrid_eckart_swfalse         112.8  110.8  111.6  110.6  110.1
hybrid_eckart_swtrue           12.6   11.9   11.5   11.4   10.8
hybrid_damped_eckart_swfalse  113.1  116.5  110.8  113.3  114.9
hybrid_damped_eckart_swtrue    12.1   10.8   10.7   11.1   11.0
```

### Fraction of trajectories whose terminating step was Newton — 10pm

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_swfalse                 0.00   0.00   0.00   0.00   0.00
hybrid_eckart_swfalse          0.00   0.00   0.00   0.00   0.00
hybrid_eckart_swtrue           0.97   0.97   0.97   0.97   0.98
hybrid_damped_eckart_swfalse   0.00   0.00   0.00   0.00   0.00
hybrid_damped_eckart_swtrue    0.98   0.98   0.98   0.98   0.98
```

## 100 pm noise — convergence %  (rows = method, columns = trust radius Å)

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_swfalse                 66.9   67.6   67.6   67.6   67.9
hybrid_eckart_swfalse           0.3    0.3    0.3    0.3    0.3
hybrid_eckart_swtrue           64.8   65.5   65.2   65.5   65.2
hybrid_damped_eckart_swfalse    0.3    0.3    0.3    0.3    0.3
hybrid_damped_eckart_swtrue    66.6   66.6   66.2   66.2   66.6
```

*GAD baselines (5k step budget, no Newton phase) for context:*

- **GAD dt=0.003 (5k)** at 100pm: conv = 71.1%, median step at conv = 756, wall/conv = 183.8 s
- **GAD dt=0.005 (5k)** at 100pm: conv = 71.8%, median step at conv = 457, wall/conv = 156.0 s
- **GAD dt=0.007 (5k)** at 100pm: conv = 72.8%, median step at conv = 331, wall/conv = 140.7 s

### Median steps to converge — 100pm  (lower is better)

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_swfalse                  540    500    484    478    480
hybrid_eckart_swfalse           916    915    915    915    915
hybrid_eckart_swtrue            194    105     60     38     33
hybrid_damped_eckart_swfalse    916    915    915    915    915
hybrid_damped_eckart_swtrue     330    173     94     49     36
```

### Wall-time per converged TS — 100pm  (sec, lower is better)

```
trust_radius                   0.005   0.010   0.020   0.050   0.100
method                                                              
hybrid_swfalse                  64.9    57.8    57.2    57.3    56.5
hybrid_eckart_swfalse        17298.7 17519.7 17362.1 17207.5 17357.5
hybrid_eckart_swtrue            46.6    40.1    40.1    36.0    36.5
hybrid_damped_eckart_swfalse 17561.9 16992.2 16854.3 16956.8 17108.3
hybrid_damped_eckart_swtrue     54.3    43.6    39.6    37.4    35.6
```

### Fraction of trajectories whose terminating step was Newton — 100pm

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_swfalse                 0.00   0.00   0.00   0.00   0.00
hybrid_eckart_swfalse          0.00   0.00   0.00   0.00   0.00
hybrid_eckart_swtrue           0.86   0.86   0.86   0.86   0.88
hybrid_damped_eckart_swfalse   0.00   0.00   0.00   0.00   0.00
hybrid_damped_eckart_swtrue    0.89   0.87   0.86   0.86   0.87
```


# Optimal hybrid GAD-Newton config per noise level


Best (method, trust_radius) per noise — head-to-head vs vanilla GAD dt=0.007 (5000-step budget):


## 10 pm noise

- **Vanilla GAD dt=0.007 baseline:** conv = 89.2% (256/287); median step at conv = 72; wall/conv = 46.6 s
- **Best hybrid by conv %:**  `hybrid_swfalse` @ trust=0.005: conv = 89.2% (256/287); median step at conv = 106; wall/conv = 16.1 s
- **Best hybrid by wall-per-conv:**  `hybrid_damped_eckart_swtrue` @ trust=0.02: conv = 86.1%; wall/conv = 10.7 s
- **Head-to-head:** hybrid is **2.9× faster per converged TS** (16.1 s vs 46.6 s); accuracy +0.0 pp

## 100 pm noise

- **Vanilla GAD dt=0.007 baseline:** conv = 72.8% (209/287); median step at conv = 331; wall/conv = 140.7 s
- **Best hybrid by conv %:**  `hybrid_swfalse` @ trust=0.1: conv = 67.9% (195/287); median step at conv = 480; wall/conv = 56.5 s
- **Best hybrid by wall-per-conv:**  `hybrid_damped_eckart_swtrue` @ trust=0.1: conv = 66.6%; wall/conv = 35.6 s
- **Head-to-head:** hybrid is **2.5× faster per converged TS** (56.5 s vs 140.7 s); accuracy -4.9 pp