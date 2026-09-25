package app_test

import (
	"context"
	"errors"
	"testing"

	"github.com/gjcourt/llmux/internal/app"
	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/testdoubles"
)

type discard struct{}

func (discard) Emit(domain.Event) error { return nil }

func TestChat_FirstMatchingProviderWins(t *testing.T) {
	specific := &testdoubles.Provider{ProviderName: "specific", ModelIDs: []string{"claude"}}
	catchAll := &testdoubles.Provider{ProviderName: "catchall"}
	svc := app.New(specific, catchAll)

	if err := svc.Chat(context.Background(), domain.ChatRequest{Model: "claude"}, discard{}); err != nil {
		t.Fatal(err)
	}
	if err := svc.Chat(context.Background(), domain.ChatRequest{Model: "qwen"}, discard{}); err != nil {
		t.Fatal(err)
	}
	if len(specific.Requests) != 1 || specific.Requests[0].Model != "claude" {
		t.Errorf("specific provider got %+v", specific.Requests)
	}
	if len(catchAll.Requests) != 1 || catchAll.Requests[0].Model != "qwen" {
		t.Errorf("catch-all got %+v", catchAll.Requests)
	}
}

func TestChat_NoProvider(t *testing.T) {
	svc := app.New(&testdoubles.Provider{ModelIDs: []string{"a"}})
	err := svc.Chat(context.Background(), domain.ChatRequest{Model: "b"}, discard{})
	if !errors.Is(err, domain.ErrNoProvider) {
		t.Fatalf("want ErrNoProvider, got %v", err)
	}
}

func TestModels_SkipsFailingProvider(t *testing.T) {
	svc := app.New(
		&testdoubles.Provider{ModelIDs: []string{"a"}},
		&testdoubles.Provider{ModelsErr: errors.New("down")},
		&testdoubles.Provider{ModelIDs: []string{"b"}},
	)
	ms, err := svc.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) != 2 || ms[0].ID != "a" || ms[1].ID != "b" {
		t.Errorf("want [a b], got %+v", ms)
	}
}
