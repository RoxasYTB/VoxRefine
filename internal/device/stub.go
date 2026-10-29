//go:build !cgo

package device

// List is unavailable without an audio backend.
func List(kind Kind) ([]Info, error) { return nil, ErrNoBackend }

// Session is a running capture → process → playback pipeline.
type Session struct{}

// Start is unavailable without an audio backend.
func Start(cfg Config, process func(in, out []float32)) (*Session, error) {
	return nil, ErrNoBackend
}

// Stop does nothing.
func (s *Session) Stop() {}

// Stats returns zeros.
func (s *Session) Stats() Stats { return Stats{} }

// Description names the devices in use.
func (s *Session) Description() string { return "" }
