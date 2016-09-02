// Copyright (c) 2016 The Decred developers.

package main

import (
	"fmt"
	"runtime"
	"time"
	"unsafe"

	"github.com/mumax/3/cuda/cu"

	"github.com/decred/gominer/util"
	"github.com/decred/gominer/work"
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

	minrLog.Infof("Started GPU #%d: %s", d.index, d.deviceName)
	outputData := make([]uint32, d.cuInSize)

	// Create the CU context
	d.cuContext = cu.CtxCreate(cu.CTX_BLOCKING_SYNC, d.cuDeviceID)

	// Allocate the input region
	d.cuContext.SetCurrent()

	N4 := int64(4 * d.cuInSize)
	d.cuInput = cu.MemAlloc(N4)

	// kernel is built with nvcc, not an api call so much bet done
	// at compile time.

	// Load the kernel and get function.
	d.cuModule = cu.ModuleLoad(cfg.CuKernel)
	d.cuKernel = d.cuModule.GetFunction("decred_gpu_hash_nonce")

	// Bump the extraNonce for the device it's running on
	// when you begin mining. This ensures each GPU is doing
	// different work. If the extraNonce has already been
	// set for valid work, restore that.
	d.extraNonce += uint32(d.index) << 24
	d.lastBlock[work.Nonce1Word] = util.Uint32EndiannessSwap(d.extraNonce)

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

		// arg 0: cleared
		outputData[0] = 0
		// args 1..8: midstate
		for i := 0; i < 8; i++ {
			outputData[i+1] = d.midstate[i]
		}
		// args 9..20: lastBlock except nonce
		i2 := 0
		for i := 0; i < 12; i++ {
			if i2 == work.Nonce0Word {
				i2++
			}
			outputData[i+9] = d.lastBlock[i2]
			i2++
		}

		cu.MemcpyHtoD(d.cuInput, unsafe.Pointer(&outputData[0]), N4)

		// Execute the kernel and follow its execution time.
		currentTime := time.Now()
		//var globalWorkSize [1]cl.CL_size_t
		//globalWorkSize[0] = cl.CL_size_t(d.workSize)
		//var localWorkSize [1]cl.CL_size_t
		//localWorkSize[0] = localWorksize

		grid := 1               // TODO
		block := 1              // TODO
		throughput := uint32(0) // TODO
		// Which nonceword is this?  In ccminer it is &pdata[35]
		nonce := d.lastBlock[work.Nonce1Word]
		targetHigh := uint32(0) // TODO
		// Provide pointer args to kernel
		args := []unsafe.Pointer{
			unsafe.Pointer(&throughput),
			unsafe.Pointer(&nonce),
			unsafe.Pointer(&d.cuInput),
			unsafe.Pointer(&targetHigh),
		}
		cu.LaunchKernel(d.cuKernel, grid, 1, 1, block, 1, 1, 0, 0, args)

		cu.MemcpyDtoH(unsafe.Pointer(&outputData[0]), d.cuInput, N4)

		for i := uint32(0); i < outputData[0]; i++ {
			minrLog.Debugf("GPU #%d: Found candidate %v nonce %08x, "+
				"extraNonce %08x, workID %08x, timestamp %08x",
				d.index, i+1, outputData[i+1], d.lastBlock[work.Nonce1Word],
				util.Uint32EndiannessSwap(d.currentWorkID),
				d.lastBlock[work.TimestampWord])

			// Assess the work. If it's below target, it'll be rejected
			// here. The mining algorithm currently sends this function any
			// difficulty 1 shares.
			d.foundCandidate(d.lastBlock[work.TimestampWord], outputData[i+1],
				d.lastBlock[work.Nonce1Word])
		}

		elapsedTime := time.Since(currentTime)
		minrLog.Tracef("GPU #%d: Kernel execution to read time: %v", d.index,
			elapsedTime)
	}

	return nil
}
