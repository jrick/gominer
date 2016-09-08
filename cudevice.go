// Copyright (c) 2016 The Decred developers.

package main

/*
#cgo LDFLAGS: -L/opt/cuda/lib64 -L/opt/cuda/lib -lcuda -lcudart obj/cuda.a
#include <stdint.h>
void decred_cpu_setBlock_52(const uint32_t *input);
*/
import "C"
import (
	"encoding/binary"
	"fmt"
	"runtime"
	"time"
	"unsafe"

	"github.com/mumax/3/cuda/cu"

	"github.com/decred/gominer/util"
	"github.com/decred/gominer/work"
)

const (
	// From ccminer
	threadsPerBlock = 640
	blockx          = threadsPerBlock
	blocky          = 1
	blockz          = 1
)

func getCUInfo() ([]cu.Device, error) {
	cu.Init(0)
	ids := cu.DeviceGetCount()
	minrLog.Infof("%v GPUs", ids)
	var CUdevices []cu.Device
	// XXX Do this more like ListCuDevices
	for i := 0; i < ids; i++ {
		dev := cu.DeviceGet(i)
		CUdevices = append(CUdevices, dev)
		minrLog.Infof("%v: %v", i, dev.Name())
	}
	return CUdevices, nil
}

// getCUDevices returns the list of devices for the given platform.
func getCUDevices() ([]cu.Device, error) {
	cu.Init(0)

	version := cu.Version()
	fmt.Println(version)

	maj := version / 1000
	min := version % 100

	minMajor := 5
	minMinor := 5

	if maj < minMajor || (maj == minMajor && min < minMinor) {
		return nil, fmt.Errorf("Driver does not suppoer CUDA %v.%v API", minMajor, minMinor)
	}

	var numDevices int
	numDevices = cu.DeviceGetCount()
	if numDevices < 1 {
		return nil, fmt.Errorf("No devices found")
	}
	devices := make([]cu.Device, numDevices)
	for i := 0; i < numDevices; i++ {
		dev := cu.DeviceGet(i)
		devices[i] = dev
	}
	return devices, nil
}

// ListCuDevices prints a list of CUDA capable GPUs present.
func ListCuDevices() {
	// CUDA devices
	// Because mumux3/3/cuda/cu likes to panic instead of error.
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("No CUDA Capable GPUs present")
		}
	}()
	devices, _ := getCUDevices()
	for i, dev := range devices {
		fmt.Printf("CUDA Capbale GPU #%d: %s\n", i, dev.Name())
	}
}

func NewCuDevice(index int, deviceID cu.Device,
	workDone chan []byte) (*Device, error) {

	d := &Device{
		index:      index,
		cuDeviceID: deviceID,
		deviceName: deviceID.Name(),
		cuda:       true,
		quit:       make(chan struct{}),
		newWork:    make(chan *work.Work, 5),
		workDone:   workDone,
	}

	d.cuInSize = 21

	d.started = uint32(time.Now().Unix())

	// Autocalibrate?

	return d, nil

}

func (d *Device) runCuDevice() error {
	// Need to have this stuff here for a ctx vs thread issue.
	runtime.LockOSThread()

	// Create the CU context
	d.cuContext = cu.CtxCreate(cu.CTX_BLOCKING_SYNC, d.cuDeviceID)

	// Allocate the input region
	d.cuContext.SetCurrent()

	// kernel is built with nvcc, not an api call so much bet done
	// at compile time.

	// Load the kernel and get function.
	d.cuModule = cu.ModuleLoad(cfg.CuKernel)
	d.cuKernel = d.cuModule.GetFunction("decred_gpu_hash_nonce")

	minrLog.Infof("Started GPU #%d: %s", d.index, d.deviceName)
	nonceResultsH := make([]uint32, d.cuInSize)
	nonceResultsD := cu.MemAlloc(d.cuInSize * 4)
	defer nonceResultsD.Free()

	const N4 = 48
	endianData := make([]byte, N4*4)

	for {
		d.updateCurrentWork()

		select {
		case <-d.quit:
			return nil
		default:
		}

		// Increment extraNonce.
		util.RolloverExtraNonce(&d.extraNonce)
		d.lastBlock[work.Nonce1Word] = util.Uint32EndiannessSwap(d.extraNonce)

		// Update the timestamp. Only solo work allows you to roll
		// the timestamp.
		ts := d.work.JobTime
		if d.work.IsGetWork {
			diffSeconds := uint32(time.Now().Unix()) - d.work.TimeReceived
			ts = d.work.JobTime + diffSeconds
		}
		d.lastBlock[work.TimestampWord] = util.Uint32EndiannessSwap(ts)

		nonceResultsH[0] = 0

		cu.MemcpyHtoD(nonceResultsD, unsafe.Pointer(&nonceResultsH[0]), d.cuInSize)

		copy(endianData, d.work.Data[:128])
		for i, j := 128, 0; i < 180; {
			b := make([]byte, 4)
			binary.LittleEndian.PutUint32(b, d.lastBlock[j])
			copy(endianData[i:], b)
			i += 4
			j++
		}
		C.decred_cpu_setBlock_52((*C.uint32_t)(unsafe.Pointer(&endianData[0])))

		// Execute the kernel and follow its execution time.
		currentTime := time.Now()

		throughput := uint32(536870912) // TODO

		//gridx := int((throughput + threadsPerBlock - 1) / threadsPerBlock) // TODO
		// ccminer uses the above which gives 838861 on my test box but
		// that fails on same machine with gominer.
		gridx := 50000
		gridy := 1
		gridz := 1
		// TODO Which nonceword is this?  In ccminer it is &pdata[35]
		nonce := d.lastBlock[work.Nonce1Word]
		targetHigh := uint32(0) // TODO

		// Provide pointer args to kernel
		args := []unsafe.Pointer{
			unsafe.Pointer(&throughput),
			unsafe.Pointer(&nonce),
			unsafe.Pointer(&nonceResultsD),
			unsafe.Pointer(&targetHigh),
		}
		cu.LaunchKernel(d.cuKernel, gridx, gridy, gridz, blockx, blocky, blockz, 0, 0, args)

		cu.MemcpyDtoH(unsafe.Pointer(&nonceResultsH[0]), nonceResultsD, d.cuInSize)

		for i := uint32(0); i < nonceResultsH[0]; i++ {
			minrLog.Debugf("%x", nonceResultsH)
			minrLog.Debugf("GPU #%d: Found candidate %v nonce %08x, "+
				"extraNonce %08x, workID %08x, timestamp %08x",
				d.index, i+1, nonceResultsH[i+1], d.lastBlock[work.Nonce1Word],
				util.Uint32EndiannessSwap(d.currentWorkID),
				d.lastBlock[work.TimestampWord])

			// Assess the work. If it's below target, it'll be rejected
			// here. The mining algorithm currently sends this function any
			// difficulty 1 shares.
			d.foundCandidate(d.lastBlock[work.TimestampWord], nonceResultsH[i+1],
				d.lastBlock[work.Nonce1Word])
		}

		elapsedTime := time.Since(currentTime)
		minrLog.Tracef("GPU #%d: Kernel execution to read time: %v", d.index,
			elapsedTime)
	}

	return nil
}
