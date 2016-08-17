// Copyright (c) 2016 The Decred developers.

package main

import (
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mumax/3/cuda/cu"

	"github.com/decred/gominer/cl"
	"github.com/decred/gominer/stratum"
	"github.com/decred/gominer/work"
)

type Miner struct {
	devices          []*ClDevice
	cudevices        []*CuDevice
	workDone         chan []byte
	quit             chan struct{}
	needsWorkRefresh chan struct{}
	wg               sync.WaitGroup
	pool             *stratum.Stratum

	started       uint32
	validShares   uint64
	staleShares   uint64
	invalidShares uint64
}

func NewMiner() (*Miner, error) {
	m := &Miner{
		workDone:         make(chan []byte, 10),
		quit:             make(chan struct{}),
		needsWorkRefresh: make(chan struct{}),
	}

	// If needed, start pool code.
	if cfg.Pool != "" && !cfg.Benchmark {
		s, err := stratum.StratumConn(cfg.Pool, cfg.PoolUser, cfg.PoolPassword, cfg.Proxy, cfg.ProxyUser, cfg.ProxyPass, version())
		if err != nil {
			return nil, err
		}
		m.pool = s
	}

	if cfg.UseCuda {
		CUdeviceIDs, err := getCUInfo()
		if err != nil {
			return nil, err
		}

		var deviceIDs []cu.Device

		// XXX Can probably combine these bits with the opencl ones once
		// I decide what to do about the types.

		// Enforce device restrictions if they exist
		if len(cfg.DeviceIDs) > 0 {
			for _, i := range cfg.DeviceIDs {
				var found = false
				for j, CUdeviceID := range CUdeviceIDs {
					if i == j {
						deviceIDs = append(deviceIDs, CUdeviceID)
						found = true
					}
				}
				if !found {
					return nil, fmt.Errorf("Unable to find GPU #%d", i)
				}
			}
		} else {
			deviceIDs = CUdeviceIDs
		}

		// Check the number of intensities/work sizes versus the number of devices.
		userSetWorkSize := true
		if reflect.DeepEqual(cfg.Intensity, defaultIntensity) &&
			reflect.DeepEqual(cfg.WorkSize, defaultWorkSize) {
			userSetWorkSize = false
		}
		if userSetWorkSize {
			if reflect.DeepEqual(cfg.WorkSize, defaultWorkSize) {
				if len(cfg.Intensity) != len(deviceIDs) {
					return nil, fmt.Errorf("Intensities supplied, but number supplied "+
						"did not match the number of GPUs (got %v, want %v)",
						len(cfg.Intensity), len(deviceIDs))
				}
			} else {
				if len(cfg.WorkSize) != len(deviceIDs) {
					return nil, fmt.Errorf("WorkSize supplied, but number supplied "+
						"did not match the number of GPUs (got %v, want %v)",
						len(cfg.WorkSize), len(deviceIDs))
				}
			}
		}
		m.devices = make([]*ClDevice, len(deviceIDs))
		for i, deviceID := range deviceIDs {
			// Use the real device order so i.e. -D 1 doesn't print GPU #0
			realnum := i
			for iCL, CLdeviceID := range CLdeviceIDs {
				if CLdeviceID == deviceID {
					realnum = iCL
				}
			}

			var err error
			m.devices[i], err = NewClDevice(realnum, platformID, deviceID, m.workDone)
			if err != nil {
				return nil, err
			}
		}
	} else {
		platformID, CLdeviceIDs, err := getCLInfo()
		if err != nil {
			return nil, err
		}

		var deviceIDs []cl.CL_device_id

		// Enforce device restrictions if they exist
		if len(cfg.DeviceIDs) > 0 {
			for _, i := range cfg.DeviceIDs {
				var found = false
				for j, CLdeviceID := range CLdeviceIDs {
					if i == j {
						deviceIDs = append(deviceIDs, CLdeviceID)
						found = true
					}
				}
				if !found {
					return nil, fmt.Errorf("Unable to find GPU #%d", i)
				}
			}
		} else {
			deviceIDs = CLdeviceIDs
		}

		// Check the number of intensities/work sizes versus the number of devices.
		userSetWorkSize := true
		if reflect.DeepEqual(cfg.Intensity, defaultIntensity) &&
			reflect.DeepEqual(cfg.WorkSize, defaultWorkSize) {
			userSetWorkSize = false
		}
		if userSetWorkSize {
			if reflect.DeepEqual(cfg.WorkSize, defaultWorkSize) {
				if len(cfg.Intensity) != len(deviceIDs) {
					return nil, fmt.Errorf("Intensities supplied, but number supplied "+
						"did not match the number of GPUs (got %v, want %v)",
						len(cfg.Intensity), len(deviceIDs))
				}
			} else {
				if len(cfg.WorkSize) != len(deviceIDs) {
					return nil, fmt.Errorf("WorkSize supplied, but number supplied "+
						"did not match the number of GPUs (got %v, want %v)",
						len(cfg.WorkSize), len(deviceIDs))
				}
			}
		}

		m.devices = make([]*ClDevice, len(deviceIDs))
		for i, deviceID := range deviceIDs {
			// Use the real device order so i.e. -D 1 doesn't print GPU #0
			realnum := i
			for iCL, CLdeviceID := range CLdeviceIDs {
				if CLdeviceID == deviceID {
					realnum = iCL
				}
			}

			var err error
			m.devices[i], err = NewClDevice(realnum, platformID, deviceID, m.workDone)
			if err != nil {
				return nil, err
			}
		}
	}
	m.started = uint32(time.Now().Unix())

	return m, nil
}

func (m *Miner) workSubmitThread() {
	defer m.wg.Done()

	for {
		select {
		case <-m.quit:
			return
		case data := <-m.workDone:
			// Only use that is we are not using a pool.
			if m.pool == nil {
				accepted, err := GetWorkSubmit(data)
				if err != nil {
					inval := atomic.LoadUint64(&m.invalidShares)
					inval++
					atomic.StoreUint64(&m.invalidShares, inval)

					minrLog.Errorf("Error submitting work: %v", err)
				} else {
					if accepted {
						val := atomic.LoadUint64(&m.validShares)
						val++
						atomic.StoreUint64(&m.validShares, val)

						minrLog.Debugf("Submitted work successfully: %v",
							accepted)
					} else {
						inval := atomic.LoadUint64(&m.invalidShares)
						inval++
						atomic.StoreUint64(&m.invalidShares, inval)
					}

					m.needsWorkRefresh <- struct{}{}
				}
			} else {
				accepted, err := GetPoolWorkSubmit(data, m.pool)
				if err != nil {
					switch err {
					case stratum.ErrStatumStaleWork:
						stale := atomic.LoadUint64(&m.staleShares)
						stale++
						atomic.StoreUint64(&m.staleShares, stale)

						minrLog.Debugf("Share submitted to pool was stale")

					default:
						inval := atomic.LoadUint64(&m.invalidShares)
						inval++
						atomic.StoreUint64(&m.invalidShares, inval)

						minrLog.Errorf("Error submitting work to pool: %v", err)
					}
				} else {
					if accepted {
						val := atomic.LoadUint64(&m.validShares)
						val++
						atomic.StoreUint64(&m.validShares, val)

						minrLog.Debugf("Submitted work to pool successfully: %v",
							accepted)
					} else {
						inval := atomic.LoadUint64(&m.invalidShares)
						inval++
						atomic.StoreUint64(&m.invalidShares, inval)

						m.invalidShares++
					}
					m.needsWorkRefresh <- struct{}{}
				}
			}
		}
	}
}

func (m *Miner) workRefreshThread() {
	defer m.wg.Done()

	t := time.NewTicker(100 * time.Millisecond)
	defer t.Stop()

	for {
		// Only use that is we are not using a pool.
		if m.pool == nil {
			work, err := GetWork()
			if err != nil {
				minrLog.Errorf("Error in getwork: %v", err)
			} else {
				for _, d := range m.devices {
					d.SetWork(work)
				}
			}
		} else {
			m.pool.Lock()
			if m.pool.PoolWork.NewWork {
				work, err := GetPoolWork(m.pool)
				m.pool.Unlock()
				if err != nil {
					minrLog.Errorf("Error in getpoolwork: %v", err)
				} else {
					for _, d := range m.devices {
						d.SetWork(work)
					}
				}
			} else {
				m.pool.Unlock()
			}
		}
		select {
		case <-m.quit:
			return
		case <-t.C:
		case <-m.needsWorkRefresh:
		}
	}
}

func (m *Miner) printStatsThread() {
	defer m.wg.Done()

	t := time.NewTicker(time.Second * 5)
	defer t.Stop()

	for {
		if cfg.Pool != "" && !cfg.Benchmark {
			valid := atomic.LoadUint64(&m.validShares)
			minrLog.Infof("Global stats: Accepted: %v, Rejected: %v, Stale: %v",
				valid,
				atomic.LoadUint64(&m.invalidShares),
				atomic.LoadUint64(&m.staleShares))

			secondsElapsed := uint32(time.Now().Unix()) - m.started
			if (secondsElapsed / 60) > 0 {
				utility := float64(valid) / (float64(secondsElapsed) / float64(60))
				minrLog.Infof("Global utility (accepted shares/min): %v", utility)
			}
		}
		for _, d := range m.devices {
			d.PrintStats()
		}

		select {
		case <-m.quit:
			return
		case <-t.C:
		case <-m.needsWorkRefresh:
		}
	}
}

func (m *Miner) Run() {
	m.wg.Add(len(m.devices))

	for _, d := range m.devices {
		device := d
		go func() {
			device.Run()
			device.Release()
			m.wg.Done()
		}()
	}

	m.wg.Add(1)
	go m.workSubmitThread()

	if cfg.Benchmark {
		minrLog.Warn("Running in BENCHMARK mode! No real mining taking place!")
		work := &work.Work{}
		for _, d := range m.devices {
			d.SetWork(work)
		}
	} else {
		m.wg.Add(1)
		go m.workRefreshThread()
	}

	m.wg.Add(1)
	go m.printStatsThread()

	m.wg.Wait()
}

func (m *Miner) Stop() {
	close(m.quit)
	for _, d := range m.devices {
		d.Stop()
		m.wg.Done()
	}
}
