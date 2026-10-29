//go:build !darwin && !linux && !windows

package deepfilter

import "errors"

func openLibrary(path string) (uintptr, error) {
	return 0, errors.New("loading the DeepFilterNet library is not supported on this platform")
}
