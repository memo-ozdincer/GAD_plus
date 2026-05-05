# Hybrid GAD-Newton sweep — pivot tables

Conv % by (method × trust_radius), per noise level:


## 10 pm noise

trust_radius                 0.005  0.010  0.020  0.050  0.100
method                                                        
hybrid_damped_eckart_swtrue   86.8   86.8   86.1   85.7   85.4
hybrid_eckart_swtrue          84.7   84.7   84.7   84.7   85.4



### Median steps to converge — 10pm:

trust_radius                 0.005  0.010  0.020  0.050  0.100
method                                                        
hybrid_damped_eckart_swtrue     35     19     10      6      5
hybrid_eckart_swtrue            21     12      8      5      5



### Fraction of last-steps that used Newton — 10pm:

trust_radius                 0.005  0.010  0.020  0.050  0.100
method                                                        
hybrid_damped_eckart_swtrue   0.98   0.98   0.98   0.98   0.98
hybrid_eckart_swtrue          0.97   0.97   0.97   0.97   0.98



## 100 pm noise

trust_radius                 0.005  0.010  0.020  0.050  0.100
method                                                        
hybrid_damped_eckart_swtrue   66.6   66.6   66.2   66.2   66.6
hybrid_eckart_swtrue          64.8   65.5   65.2   65.5   65.2
hybrid_swfalse                 NaN    NaN   67.6   67.6   67.9



### Median steps to converge — 100pm:

trust_radius                 0.005  0.010  0.020  0.050  0.100
method                                                        
hybrid_damped_eckart_swtrue    330    173     94     49     36
hybrid_eckart_swtrue           194    105     60     38     33
hybrid_swfalse                 NaN    NaN    484    478    480



### Fraction of last-steps that used Newton — 100pm:

trust_radius                 0.005  0.010  0.020  0.050  0.100
method                                                        
hybrid_damped_eckart_swtrue   0.89   0.87   0.86   0.86   0.87
hybrid_eckart_swtrue          0.86   0.86   0.86   0.86   0.88
hybrid_swfalse                 NaN    NaN   0.00   0.00   0.00

