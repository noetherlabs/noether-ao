# Measured latency

All units ms except compile_s and miss fractions. Host roundtrip includes forced copies.

| mode | n | wavelengths | precision | engine | path | p50_ms | p90_ms | p99_ms | p99.9_ms | max_ms | miss_1ms | miss_0.5ms | compile_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu_all | 32 | 1 | float64 | dlux | device | 0.036125 | 0.0459211 | 0.0866283 | 0.195971 | 0.3535 | 0 | 0 | 0.738152 |
| cpu_all | 32 | 1 | float64 | dlux | host_roundtrip | 0.100084 | 0.118833 | 0.266626 | 0.521254 | 0.775 | 0 | 0.0012 | 0.738152 |
| cpu_all | 32 | 1 | float64 | jax | device | 0.0365 | 0.045166 | 0.0709178 | 0.155585 | 0.21975 | 0 | 0 | 0.0935268 |
| cpu_all | 32 | 1 | float64 | jax | host_roundtrip | 0.100916 | 0.104166 | 0.126167 | 0.203178 | 0.42275 | 0 | 0 | 0.0935268 |
