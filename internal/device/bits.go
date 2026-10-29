package device

import "math"

func float32Bits(f float32) uint32     { return math.Float32bits(f) }
func float32FromBits(b uint32) float32 { return math.Float32frombits(b) }
