// Copyright (c) 2016 The Decred developers.

package main

import (
	"github.com/mumax/3/cuda/cu"
)

func getCUDeviceIDs() (int, error) {
	cu.Init(0)
	ids := cu.DeviceGetCount()
	minrLog.Infof("%v GPUs", ids)
	for i := 0; i < ids; i++ {
		dev := cu.DeviceGet(i)
		minrLog.Infof("%v: %v", i, dev.Name())
	}
	return ids, nil
}
