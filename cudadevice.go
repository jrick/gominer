// Copyright (c) 2016 The Decred developers.

package main

import (
	"github.com/mumax/3/cuda/cu"
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
