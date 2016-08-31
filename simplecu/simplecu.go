package main

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
)

func main() {
	cu.Init(0)

	//num := cu.DeviceGetCount()

	//for i := 0; i < num; i++ {
	//	go run(i)
	//}

	go run(1)

	time.Sleep(5 * time.Second)

	return
}

func run(i int) {
	var midstate [8]uint32
	var lastBlock [16]uint32
	Nonce0Word := 3

	midstate[0] = 3787949510
	midstate[1] = 3207530628
	midstate[2] = 1111317365
	midstate[3] = 1138308834
	midstate[4] = 1352282114
	midstate[5] = 363263180
	midstate[6] = 2723723386
	midstate[7] = 1846442091

	lastBlock[0] = 3488022528
	lastBlock[1] = 2100625408
	lastBlock[2] = 2515125847
	lastBlock[3] = 0
	lastBlock[4] = 16777216
	lastBlock[5] = 3808506336
	lastBlock[6] = 0
	lastBlock[7] = 0
	lastBlock[8] = 0
	lastBlock[9] = 0
	lastBlock[10] = 0
	lastBlock[11] = 0

	time.Sleep(1 * time.Second)
	fmt.Printf("Running on GPU: %v\n", i)
	dev := cu.DeviceGet(i)
	ctx := cu.CtxCreate(cu.CTX_BLOCKING_SYNC, dev)
	ctx.SetCurrent()

	mod := cu.ModuleLoad("decred.ptx")
	f := mod.GetFunction("decred_gpu_hash_nonce")

	N := 21
	a := make([]uint32, N)

	// args 1..8: midstate
	for i := 0; i < 8; i++ {
		a[i] = midstate[i]
	}
	// args 9..20: lastBlock except nonce
	i2 := 0
	for i := 0; i < 12; i++ {
		if i2 == Nonce0Word {
			i2++
		}
		a[i+9] = lastBlock[i2]
		i2++
	}

	N4 := int64(unsafe.Sizeof(a[0])) * int64(N)

	A := cu.MemAlloc(N4)
	defer A.Free()
	aptr := unsafe.Pointer(&a[0])
	cu.MemcpyHtoD(A, aptr, N4)

	block := 128
	grid := DivUp(N, block)
	shmem := 0

	maxResults := int64(4)

	resSize := maxResults * int64(unsafe.Sizeof(a[0]))

	dResNonce := cu.MemAlloc(resSize)
	hResNonce := cu.MemAllocHost(resSize)

	//(throughput, (*pnonce), d_resNonce[thr_id], targetHigh)

	args := []unsafe.Pointer{unsafe.Pointer(&A), unsafe.Pointer(&dResNonce)}
	cu.LaunchKernel(f, grid, 1, 1, block, 1, 1, shmem, 0, args)

	cu.MemcpyDtoH(hResNonce, dResNonce, resSize)

}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}
