// Copyright (c) 2016 The Decred developers.

package main

import (
	"fmt"

	"github.com/mumax/3/cuda/cu"

	"github.com/decred/gominer/util"
	"github.com/decred/gominer/work"
)

func getCUInfo() ([]cu.Device, error) {
	cu.Init(0)
	// XXX check cudaDriverGetVersion?
	ids := cu.DeviceGetCount()
	minrLog.Infof("%v GPUs", ids)
	var CUdevices []cu.Device
	// XXX Do this more like ListCuDevices
	for i := 0; i < ids; i++ {
		dev := cu.DeviceGet(i)
		CUdevices = append(CUdevices, dev)
		minrLog.Infof("%v: %v", i, dev.Name())
		// XXX check cudaGetDeviceProperties?
	}
	return CUdevices, nil
}

// getCUDevices returns the list of devices for the given platform.
func getCUDevices() ([]cu.Device, error) {
	cu.Init(0)
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

	// Create tue CU context
	d.cuContext = cu.CtxCreate(cu.CTX_SCHED_AUTO, deviceID)

	// kernel is built with nvcc, not an api call so much bet done
	// at compile time.

	// Load the kernel and get function.
	mod := cu.ModuleLoad(cfg.CuKernel)
	f := mod.GetFunction("hash")

	// Autocalibrate?

	return d, nil

}

func (d *Device) runCuDevice() error {
	minrLog.Infof("Started GPU #%d: %s", d.index, d.deviceName)
	//outputData := make([]uint32, outputBufferSize)

	// Set the current context
	cu.CtxSetCurrent(d.cuContext)

	// Bump the extraNonce for the device it's running on
	// when you begin mining. This ensures each GPU is doing
	// different work. If the extraNonce has already been
	// set for valid work, restore that.
	d.extraNonce += uint32(d.index) << 24
	d.lastBlock[work.Nonce1Word] = util.Uint32EndiannessSwap(d.extraNonce)

	return nil
}
