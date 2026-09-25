package main

import "testing"

// A backend is disabled by setting its URL to the empty string. The original
// getter treated "" as unset and fell back to the default, so this never
// worked; AGENTS.md documented the intended behaviour.
func TestEnvOr_EmptyDisables(t *testing.T) {
	t.Setenv("LLMUX_TEST_URL", "")
	if got := envOr("LLMUX_TEST_URL", "http://default"); got != "" {
		t.Errorf("set-but-empty must return empty, got %q", got)
	}
}

func TestEnvOr_UnsetFallsBack(t *testing.T) {
	if got := envOr("LLMUX_TEST_URL_NEVER_SET", "http://default"); got != "http://default" {
		t.Errorf("unset must fall back, got %q", got)
	}
}

func TestEnvOr_SetWins(t *testing.T) {
	t.Setenv("LLMUX_TEST_URL", "http://custom")
	if got := envOr("LLMUX_TEST_URL", "http://default"); got != "http://custom" {
		t.Errorf("got %q", got)
	}
}
