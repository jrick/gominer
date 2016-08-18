// Copyright (c) 2016 The Decred developers.

package main

import (
	"fmt"

	"github.com/decred/gominer/cl"
)

func getCLInfo() (cl.CL_platform_id, []cl.CL_device_id, error) {
	var platformID cl.CL_platform_id
	platformIDs, err := getCLPlatforms()
	if err != nil {
		return platformID, nil, fmt.Errorf("Could not get CL platforms: %v", err)
	}
	platformID = platformIDs[0]
	CLdeviceIDs, err := getCLDevices(platformID)
	if err != nil {
		return platformID, nil, fmt.Errorf("Could not get CL devices for platform: %v", err)
	}
	return platformID, CLdeviceIDs, nil
}

func getCLPlatforms() ([]cl.CL_platform_id, error) {
	var numPlatforms cl.CL_uint
	status := cl.CLGetPlatformIDs(0, nil, &numPlatforms)
	if status != cl.CL_SUCCESS {
		return nil, clError(status, "CLGetPlatformIDs")
	}
	platforms := make([]cl.CL_platform_id, numPlatforms)
	status = cl.CLGetPlatformIDs(numPlatforms, platforms, nil)
	if status != cl.CL_SUCCESS {
		return nil, clError(status, "CLGetPlatformIDs")
	}
	return platforms, nil
}

// getCLDevices returns the list of devices for the given platform.
func getCLDevices(platform cl.CL_platform_id) ([]cl.CL_device_id, error) {
	var numDevices cl.CL_uint
	status := cl.CLGetDeviceIDs(platform, cl.CL_DEVICE_TYPE_GPU, 0, nil,
		&numDevices)
	if status != cl.CL_SUCCESS {
		return nil, clError(status, "CLGetDeviceIDs")
	}
	devices := make([]cl.CL_device_id, numDevices)
	status = cl.CLGetDeviceIDs(platform, cl.CL_DEVICE_TYPE_ALL, numDevices,
		devices, nil)
	if status != cl.CL_SUCCESS {
		return nil, clError(status, "CLGetDeviceIDs")
	}
	return devices, nil
}
