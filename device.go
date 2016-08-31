// Copyright (c) 2016 The Decred developers.

package main

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"math/big"
	"os"
	"sync"
	"time"
	"unsafe"

	"github.com/mumax/3/cuda/cu"

	"github.com/decred/dcrd/blockchain"
	"github.com/decred/dcrd/chaincfg"
	"github.com/decred/dcrd/chaincfg/chainhash"

	"github.com/decred/gominer/blake256"
	"github.com/decred/gominer/cl"
	"github.com/decred/gominer/util"
	"github.com/decred/gominer/work"
)

const (
	outputBufferSize   = cl.CL_size_t(64)
	localWorksize      = 64
	uint32Size         = cl.CL_size_t(unsafe.Sizeof(cl.CL_uint(0)))
	cuOutputBufferSize = 64
)

var chainParams = &chaincfg.MainNetParams

var zeroSlice = []cl.CL_uint{cl.CL_uint(0)}

func loadProgramSource(filename string) ([][]byte, []cl.CL_size_t, error) {
	var program_buffer [1][]byte
	var program_size [1]cl.CL_size_t

	// Read each program file and place content into buffer array.
	program_handle, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer program_handle.Close()

	buf := bytes.NewBuffer(nil)
	_, err = io.Copy(buf, program_handle)
	if err != nil {
		return nil, nil, err
	}
	str := string(buf.Bytes())
	program_final := []byte(str)

	program_size[0] = cl.CL_size_t(len(program_final))
	program_buffer[0] = make([]byte, program_size[0])
	for i := range program_final {
		program_buffer[0][i] = program_final[i]
	}

	return program_buffer[:], program_size[:], nil
}

type Device struct {
	sync.Mutex
	index int
	cuda  bool

	// Items for OpenCL device
	platformID   cl.CL_platform_id
	deviceID     cl.CL_device_id
	deviceName   string
	context      cl.CL_context
	queue        cl.CL_command_queue
	outputBuffer cl.CL_mem
	program      cl.CL_program
	kernel       cl.CL_kernel

	// Items for CUDA device
	cuDeviceID     cu.Device
	cuContext      cu.Context
	cuModule       cu.Module
	cuKernel       cu.Function
	cuInput        cu.DevicePtr
	cuOutputBuffer []float64

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

func clError(status cl.CL_int, f string) error {
	if -status < 0 || int(-status) > len(cl.ERROR_CODES_STRINGS) {
		return fmt.Errorf("%s returned unknown error!")
	}

	return fmt.Errorf("%s returned error %s (%d)", f,
		cl.ERROR_CODES_STRINGS[-status], status)
}

func (d *Device) Release() {
	if d.cuda {
		d.cuContext.SetCurrent()
		d.cuInput.Free()
		cu.CtxDestroy(&d.cuContext)
	} else {
		cl.CLReleaseKernel(d.kernel)
		cl.CLReleaseProgram(d.program)
		cl.CLReleaseCommandQueue(d.queue)
		cl.CLReleaseMemObject(d.outputBuffer)
		cl.CLReleaseContext(d.context)
	}
}

func (d *Device) updateCurrentWork() {
	var w *work.Work
	if d.hasWork {
		// If we already have work, we just need to check if there's new one
		// without blocking if there's not.
		select {
		case w = <-d.newWork:
		default:
			return
		}
	} else {
		// If we don't have work, we block until we do. We need to watch for
		// quit events too.
		select {
		case w = <-d.newWork:
		case <-d.quit:
			return
		}
	}

	d.hasWork = true

	d.work = *w
	minrLog.Tracef("pre-nonce: %v", hex.EncodeToString(d.work.Data[:]))

	// Bump and set the work ID if the work is new.
	d.currentWorkID++
	binary.LittleEndian.PutUint32(d.work.Data[128+4*work.Nonce2Word:],
		d.currentWorkID)

	// Reset the hash state
	copy(d.midstate[:], blake256.IV256[:])

	// Hash the two first blocks
	blake256.Block(d.midstate[:], d.work.Data[0:64], 512)
	blake256.Block(d.midstate[:], d.work.Data[64:128], 1024)
	minrLog.Tracef("midstate input data for work update %v",
		hex.EncodeToString(d.work.Data[0:128]))

	// Convert the next block to uint32 array.
	for i := 0; i < 16; i++ {
		d.lastBlock[i] = binary.BigEndian.Uint32(d.work.Data[128+i*4 : 132+i*4])
	}
	minrLog.Tracef("work data for work update: %v",
		hex.EncodeToString(d.work.Data[:]))
}

func (d *Device) Run() {
	var err error
	if d.cuda {
		err = d.runCuDevice()
	} else {
		err = d.runDevice()
	}
	if err != nil {
		minrLog.Errorf("Error on device: %v", err)
	}
}

// testFoundCandidate has some hardcoded data to match up with sgminer.
func (d *Device) testFoundCandidate() {
	n1 := uint32(33554432)
	n0 := uint32(7245027)

	d.midstate[0] = uint32(2421507776)
	d.midstate[1] = uint32(2099684366)
	d.midstate[2] = uint32(8033620)
	d.midstate[3] = uint32(950943511)
	d.midstate[4] = uint32(2489053653)
	d.midstate[5] = uint32(3357747798)
	d.midstate[6] = uint32(2534384973)
	d.midstate[7] = uint32(2947973092)

	target, _ := hex.DecodeString("00000000ffff0000000000000000000000000000000000000000000000000000")
	bigTarget := new(big.Int)
	bigTarget.SetString(hex.EncodeToString(target), 16)
	d.work.Target = bigTarget

	data, _ := hex.DecodeString("01000000509a3b7c65f8986a464c0e82ec5ca6aaf18cf13787507cbfc20a000000000000a455f69725e9c8623baa3c9c5a708aefb947702dc2b620b4c10129977e104c0275571a5ca5b1308b075fe74224504c9e6b1153f3de97235e7a8c7e58ea8f1c55010086a1d41fb3ee05000000fda400004a33121a2db33e1101000000abae0000260800008ec78357000000000000000000a461f2e3014335000000000000000000000000000000000000000000000000000000000000000000000000")
	copy(d.work.Data[:], data)

	minrLog.Errorf("data: %v", d.work.Data)
	minrLog.Errorf("target: %v", d.work.Target)
	minrLog.Errorf("nonce1 %x, nonce0: %x", n1, n0)

	// d.foundCandidate(n1, n0, ts)

	//need to match
	//00000000df6ffb6059643a9215f95751baa7b1ed8aa93edfeb9a560ecb1d5884
	//stratum submit {"params": ["test", "76df", "0200000000a461f2e3014335", "5783c78e", "e38c6e00"], "id": 4, "method": "mining.submit"}
}

func (d *Device) foundCandidate(ts, nonce0, nonce1 uint32) {
	d.Lock()
	defer d.Unlock()
	// Construct the final block header.
	data := make([]byte, 192)
	copy(data, d.work.Data[:])

	binary.BigEndian.PutUint32(data[128+4*work.TimestampWord:], ts)
	binary.BigEndian.PutUint32(data[128+4*work.Nonce0Word:], nonce0)
	binary.BigEndian.PutUint32(data[128+4*work.Nonce1Word:], nonce1)
	hash := chainhash.HashFuncH(data[0:180])

	// Hashes that reach this logic and fail the minimal proof of
	// work check are considered to be hardware errors.
	hashNum := blockchain.ShaHashToBig(&hash)
	if hashNum.Cmp(chainParams.PowLimit) > 0 {
		minrLog.Errorf("GPU #%d: Hardware error found, hash %v above "+
			"minimum target %032x", d.index, hash, d.work.Target.Bytes())
		d.invalidShares++
		return
	} else {
		d.allDiffOneShares++
	}

	if !cfg.Benchmark {
		// Assess versus the pool or daemon target.
		if hashNum.Cmp(d.work.Target) > 0 {
			minrLog.Debugf("GPU #%d: Hash %v bigger than target %032x (boo)",
				d.index, hash, d.work.Target.Bytes())
		} else {
			minrLog.Infof("GPU #%d: Found hash with work below target! %v (yay)",
				d.index, hash)
			d.validShares++
			d.workDone <- data
		}
	}
}

func (d *Device) Stop() {
	close(d.quit)
}

func (d *Device) SetWork(w *work.Work) {
	d.newWork <- w
}

func getDeviceInfo(id cl.CL_device_id,
	name cl.CL_device_info,
	str string) string {

	var errNum cl.CL_int
	var paramValueSize cl.CL_size_t

	errNum = cl.CLGetDeviceInfo(id, name, 0, nil, &paramValueSize)

	if errNum != cl.CL_SUCCESS {
		return fmt.Sprintf("Failed to find OpenCL device info %s.\n", str)
	}

	var info interface{}
	errNum = cl.CLGetDeviceInfo(id, name, paramValueSize, &info, nil)
	if errNum != cl.CL_SUCCESS {
		return fmt.Sprintf("Failed to find OpenCL device info %s.\n", str)
	}

	strinfo := fmt.Sprintf("%v", info)

	return strinfo
}

func (d *Device) PrintStats() {
	secondsElapsed := uint32(time.Now().Unix()) - d.started
	if secondsElapsed == 0 {
		return
	}

	diffOneShareHashesAvg := uint64(0x00000000FFFFFFFF)
	d.Lock()
	defer d.Unlock()
	averageHashRate := (float64(diffOneShareHashesAvg) *
		float64(d.allDiffOneShares)) /
		float64(secondsElapsed)

	minrLog.Infof("GPU #%d (%s) reporting average hash rate %v, %v/%v valid work",
		d.index,
		d.deviceName,
		util.FormatHashRate(averageHashRate),
		d.validShares,
		d.validShares+d.invalidShares)
}
