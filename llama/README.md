# `llama`

This package integrates llama.cpp as an Ollama runner.

### AVX

```shell
go build -tags avx .
```

### AVX2

```shell
# go doesn't recognize `-mfma` as a valid compiler flag
# see https://github.com/golang/go/issues/17895
go env -w "CGO_CFLAGS_ALLOW=-mfma"
go env -w "CGO_CXXFLAGS_ALLOW=-mfma"
go build -tags=avx2 .
```

### CUDA

```shell
go build -tags=cuda .
```

### ROCm (todo)

```shell
go build -tags=rocm .
```
