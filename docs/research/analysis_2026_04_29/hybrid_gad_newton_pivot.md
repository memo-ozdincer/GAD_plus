# Hybrid GAD-Newton sweep — pivot tables


## 10 pm noise — convergence %  (rows = method, columns = trust radius Å)

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_eckart_swfalse          45.6   46.3   46.3   46.3   46.3
hybrid_eckart_swtrue           84.7   84.7   84.7   84.7   85.4
hybrid_damped_eckart_swfalse   45.6   46.3   46.3   46.3   46.3
hybrid_damped_eckart_swtrue    86.1   86.1   86.1   85.4   85.4
```

*GAD baselines (5k step budget, no Newton phase) for context:*

- **GAD dt=0.003 (5k)** at 10pm: conv = 89.2%, median step at conv = 164, wall/conv = 54.2 s
- **GAD dt=0.005 (5k)** at 10pm: conv = 89.2%, median step at conv = 98, wall/conv = 47.7 s
- **GAD dt=0.007 (5k)** at 10pm: conv = 89.2%, median step at conv = 72, wall/conv = 46.6 s

### Median steps to converge — 10pm  (lower is better)

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_eckart_swfalse           702    702    702    702    702
hybrid_eckart_swtrue             21     12      8      5      5
hybrid_damped_eckart_swfalse    702    702    702    702    702
hybrid_damped_eckart_swtrue      19     11      7      5      4
```

### Wall-time per converged TS — 10pm  (sec, lower is better)

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_eckart_swfalse         112.7  114.5  111.0  110.8  110.4
hybrid_eckart_swtrue           12.7   12.2   11.7   11.3   10.7
hybrid_damped_eckart_swfalse  117.8  112.8  112.7  112.3  111.8
hybrid_damped_eckart_swtrue    11.5   10.9   10.3   10.8   11.3
```

### Fraction of trajectories whose terminating step was Newton — 10pm

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_eckart_swfalse          0.00   0.00   0.00   0.00   0.00
hybrid_eckart_swtrue           0.97   0.97   0.97   0.97   0.98
hybrid_damped_eckart_swfalse   0.00   0.00   0.00   0.00   0.00
hybrid_damped_eckart_swtrue    0.98   0.98   0.98   0.98   0.98
```

## 100 pm noise — convergence %  (rows = method, columns = trust radius Å)

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_eckart_swfalse           0.3    0.3    0.3    0.3    0.3
hybrid_eckart_swtrue           64.8   65.5   65.2   65.5   65.2
hybrid_damped_eckart_swfalse    0.3    0.3    0.3    0.3    0.3
hybrid_damped_eckart_swtrue    66.9   65.9   66.2   66.9   66.6
```

*GAD baselines (5k step budget, no Newton phase) for context:*

- **GAD dt=0.003 (5k)** at 100pm: conv = 71.1%, median step at conv = 756, wall/conv = 183.8 s
- **GAD dt=0.005 (5k)** at 100pm: conv = 71.8%, median step at conv = 457, wall/conv = 156.0 s
- **GAD dt=0.007 (5k)** at 100pm: conv = 72.8%, median step at conv = 331, wall/conv = 140.7 s

### Median steps to converge — 100pm  (lower is better)

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_eckart_swfalse           916    915    915    915    915
hybrid_eckart_swtrue            195    105     60     38     33
hybrid_damped_eckart_swfalse    916    915    915    915    915
hybrid_damped_eckart_swtrue     200    103     59     36     31
```

### Wall-time per converged TS — 100pm  (sec, lower is better)

```
trust_radius                   0.005   0.010   0.020   0.050   0.100
method                                                              
hybrid_eckart_swfalse        17065.6 16897.2 16933.4 17088.6 17293.2
hybrid_eckart_swtrue            47.1    41.0    37.9    35.9    36.3
hybrid_damped_eckart_swfalse 16876.5 17093.7 17011.0 16663.7 17060.5
hybrid_damped_eckart_swtrue     45.1    40.1    36.8    34.7    34.2
```

### Fraction of trajectories whose terminating step was Newton — 100pm

```
trust_radius                  0.005  0.010  0.020  0.050  0.100
method                                                         
hybrid_eckart_swfalse          0.00   0.00   0.00   0.00   0.00
hybrid_eckart_swtrue           0.86   0.86   0.86   0.86   0.88
hybrid_damped_eckart_swfalse   0.00   0.00   0.00   0.00   0.00
hybrid_damped_eckart_swtrue    0.87   0.86   0.86   0.87   0.86
```


# Optimal hybrid GAD-Newton config per noise level


Best (method, trust_radius) per noise — head-to-head vs vanilla GAD dt=0.007 (5000-step budget):


## 10 pm noise

- **Vanilla GAD dt=0.007 baseline:** conv = 89.2% (256/287); median step at conv = 72; wall/conv = 46.6 s
- **Best hybrid by conv %:**  `hybrid_damped_eckart_swtrue` @ trust=0.005: conv = 86.1% (247/287); median step at conv = 19; wall/conv = 11.5 s
- **Head-to-head:** hybrid is **4.0× faster per converged TS** (11.5 s vs 46.6 s); accuracy -3.1 pp

## 100 pm noise

- **Vanilla GAD dt=0.007 baseline:** conv = 72.8% (209/287); median step at conv = 331; wall/conv = 140.7 s
- **Best hybrid by conv %:**  `hybrid_damped_eckart_swtrue` @ trust=0.005: conv = 66.9% (192/287); median step at conv = 200; wall/conv = 45.1 s
- **Head-to-head:** hybrid is **3.1× faster per converged TS** (45.1 s vs 140.7 s); accuracy -5.9 pp