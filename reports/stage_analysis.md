# Historical stage analysis (completed shipments, synthetic data)
Descriptive associations only; never used as model inputs. Columns prefixed `_latent_` are simulator audit variables.

## stage_delays_by_destination

```
                   n port_processing_delay_h        customs_delay_h        inland_delivery_delay_h        downstream_delay_h
                                        mean median            mean median                    mean median               mean median
destination_port
Dubai             42                    8.54   4.38            2.98   0.25                    1.18  -0.34              12.70   6.42
Hamburg           42                   12.51   7.17           12.26  -0.06                    1.72  -0.43              26.50  19.75
Los Angeles       42                   17.37  12.54           14.48   4.58                    2.39  -0.08              34.23  25.16
New York          41                   18.12  13.78            7.52   2.95                    1.39  -0.25              27.04  21.58
Rotterdam         42                   15.73  10.35           16.90   6.06                    0.82  -0.32              33.44  20.26
Singapore         41                    8.43   4.75            4.85   0.89                    0.82  -0.42              14.11   3.95
```

## port_delay_by_congestion

```
                       count   mean  median
dest_congestion_index
<0.30                     44   1.58   -0.52
0.30-0.45                 85  11.85    5.81
0.45-0.60                 71  20.00   14.03
>=0.60                    25  25.06   19.52
missing                   25   9.60    8.82
```

## port_delay_by_arrival_status

```
                 count   mean  median
arrival_delay_h
early (<0h)         65   5.54    4.69
0-12h late          85   2.66    1.82
12-24h late         59  29.41   27.31
>24h late           41  25.41   22.19
```

## customs_delay_by_documentation

```
                        count   mean  median  inspection_rate
documentation_complete
False                      40  41.89   38.28            0.275
True                      210   3.76    0.40            0.081
```

## truck_wait_by_congestion

```
                          count  mean  median  zero_wait_share
_latent_congestion_dwell
<0.30                        25  0.91    0.00            0.800
0.30-0.45                    46  1.91    0.00            0.587
0.45-0.60                    53  4.03    0.00            0.642
>=0.60                       37  5.09    1.03            0.432
```

## upstream_vs_downstream_means

```
                  n  port_processing_delay_h  customs_delay_h  inland_delivery_delay_h  downstream_delay_h
arrival_delay_h
early (<0h)      65                     5.54            14.63                     1.92               22.09
0-12h late       85                     2.66             6.90                     0.66               10.22
12-24h late      59                    29.41             5.01                     2.47               36.88
>24h late        41                    25.41            15.41                     0.52               41.34
```

## upstream_vs_downstream_spearman

```
port_processing_delay_h    0.434
customs_delay_h           -0.020
inland_delivery_delay_h   -0.115
downstream_delay_h         0.247
```
