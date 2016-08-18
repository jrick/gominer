// Copyright (c) 2016 The Decred developers.

package main

import (
	"fmt"

	"github.com/mumax/3/cuda/cu"

	"github.com/decred/gominer/work"
)

func getCUInfo() ([]cu.Device, error) {
	cu.Init(0)
	// XXX check cudaDriverGetVersion?
	ids := cu.DeviceGetCount()
	minrLog.Infof("%v GPUs", ids)
	var CUdevices []cu.Device
	for i := 0; i < ids; i++ {
		dev := cu.DeviceGet(i)
		ctx := cu.CtxCreate(cu.CTX_SCHED_AUTO, dev)
		cu.CtxSetCurrent(ctx)
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
		index: index,
		//platformID: platformID,
		//deviceID:   deviceID,
		//deviceName: getDeviceInfo(deviceID, cl.CL_DEVICE_NAME, "CL_DEVICE_NAME"),
		cuda:     true,
		quit:     make(chan struct{}),
		newWork:  make(chan *work.Work, 5),
		workDone: workDone,
	}

	// Create tue CU context

	return d, nil

}
