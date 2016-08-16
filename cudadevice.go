// Copyright (c) 2016 The Decred developers.

package main

import (
	"sync"

	"github.com/decred/gominer/cl"
	"github.com/decred/gominer/work"
	"github.com/mumax/3/cuda/cu"
)

func getCUInfo() (int, error) {
	cu.Init(0)
	ids := cu.DeviceGetCount()
	minrLog.Infof("%v GPUs", ids)
	for i := 0; i < ids; i++ {
		dev := cu.DeviceGet(i)
		minrLog.Infof("%v: %v", i, dev.Name())
		ctx := cu.CtxCreate(cu.CTX_SCHED_AUTO, dev)
		cu.CtxSetCurrent(ctx)
	}
	return ids, nil
}

type CUDevice struct {
	sync.Mutex
	index        int
	platformID   cl.CL_platform_id
	deviceID     cl.CL_device_id
	deviceName   string
	context      cl.CL_context
	queue        cl.CL_command_queue
	outputBuffer cl.CL_mem
	program      cl.CL_program
	kernel       cl.CL_kernel

	workSize uint32

	// extraNonce is the device extraNonce, where the first
	// byte is the device ID (supporting up to 255 devices)
	// while the last 3 bytes is the extraNonce value. If
	// the extraNonce goes through all 0x??FFFFFF values,
	// it will reset to 0x??000000.
	extraNonce    uint32
	currentWorkID uint32

	midstate  [8]uint32
	lastBlock [16]uint32

	work     work.Work
	newWork  chan *work.Work
	workDone chan []byte
	hasWork  bool

	started          uint32
	allDiffOneShares uint64
	validShares      uint64
	invalidShares    uint64

	quit chan struct{}
}
