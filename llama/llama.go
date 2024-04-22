package llama

// #cgo darwin,arm64 CFLAGS: -std=c11 -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo darwin,arm64 CXXFLAGS: -std=c++11 -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo darwin,amd64 CXXFLAGS: -std=c++11
// #cgo darwin,arm64 LDFLAGS: ggml-metal.o -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
// #cgo darwin,amd64 LDFLAGS: -framework Foundation -framework Accelerate
// #cgo avx CFLAGS: -mavx
// #cgo avx CXXFLAGS: -mavx
// #cgo avx2 CFLAGS: -mavx -mavx2 -mfma
// #cgo avx2 CXXFLAGS: -mavx -mavx2 -mfma
// #cgo avx2 LDFLAGS: -lm
// #cgo cuda CFLAGS: -DGGML_USE_CUDA -DGGML_SHARED -mavx
// #cgo cuda CXXFLAGS: -std=c++11 -DGGML_USE_CUDA -DGGML_SHARED -mavx
// #cgo cuda LDFLAGS: -L. -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64" -lggml-cuda -lcuda -lcudart -lcublas -lcublasLt
// #include <stdlib.h>
// #include "llama.h"
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// SystemInfo is an unused example of calling llama.cpp functions using CGo
func SystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}

func Run(modelPath string, prompt string) {
	C.llama_backend_init()

	params := C.llama_model_default_params()
	model := C.llama_load_model_from_file(C.CString(modelPath), params)

	ctxParams := C.llama_context_default_params()
	ctxParams.seed = 1234
	ctxParams.n_ctx = 2048
	ctxParams.n_threads = C.uint(runtime.NumCPU())
	ctxParams.n_threads_batch = ctxParams.n_threads

	llamaContext := C.llama_new_context_with_model(model, ctxParams)
	if llamaContext == nil {
		panic("Failed to create context")
	}

	tokens, err := tokenize(model, prompt, 2048, false, false)
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Println("Tokens:", tokens)
	}

	// sequenceLength := 32

	// nCtx := C.llama_n_ctx(llamaContext)

	for _, t := range tokens {
		fmt.Println(llamaTokenToPiece(model, t))
	}

	// batch := C.llama_batch_init(512, 0, 1)

	// prompt eval
	// 	for _, t := range tokens {
	// 		C.llama_batch_add(batch, C.int32(t), i, { 0 }, false);
	// 	}
}

func llamaTokenToPiece(model *C.struct_llama_model, token int) string {
	buf := make([]byte, 12)

	// Call the C function with appropriate conversions.
	C.llama_token_to_piece(
		model,
		C.int32_t(token),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(12),
	)

	return string(buf)
}

func tokenize(model *C.struct_llama_model, text string, maxTokens int, addSpecial bool, parseSpecial bool) ([]int, error) {
	cTokens := make([]C.llama_token, maxTokens)
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.llama_tokenize(
		model,
		cText,
		C.int32_t(len(text)),
		&cTokens[0],
		C.int32_t(maxTokens),
		C.bool(addSpecial),
		C.bool(parseSpecial),
	)

	if result < 0 {
		return nil, fmt.Errorf("tokenization failed, required %d tokens", -result)
	}

	tokens := make([]int, int(result))
	for i := 0; i < int(result); i++ {
		tokens[i] = int(cTokens[i])
	}

	return tokens, nil
}
