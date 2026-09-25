package httpapi_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gjcourt/llmux/internal/adapters/httpapi"
	"github.com/gjcourt/llmux/internal/app"
	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/testdoubles"
)

const (
	webKey    = "web-key-0123456789abcdef0123456789abcdef"
	reviewKey = "review-key-0123456789abcdef0123456789ab"
)

func authServer(t *testing.T, keys map[string]string) (*httptest.Server, *testdoubles.Provider) {
	t.Helper()
	p := &testdoubles.Provider{Events: []domain.Event{{Kind: domain.EventStart}, {Kind: domain.EventText, Text: "hi"}, {Kind: domain.EventFinish, FinishReason: "stop"}}}
	srv := httptest.NewServer(httpapi.New(app.New(p), httpapi.WithClientKeys(keys)))
	t.Cleanup(srv.Close)
	return srv, p
}

func do(t *testing.T, method, url, body string, headers map[string]string) int {
	t.Helper()
	req, _ := http.NewRequest(method, url, strings.NewReader(body))
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	return resp.StatusCode
}

const chatBody = `{"model":"m","messages":[{"role":"user","content":"x"}]}`

func TestAuth_OffByDefault(t *testing.T) {
	srv, p := authServer(t, nil)
	if code := do(t, "POST", srv.URL+"/v1/chat/completions", chatBody, nil); code != 200 {
		t.Fatalf("status %d", code)
	}
	if p.Requests[0].Client != httpapi.Anonymous {
		t.Errorf("client = %q, want anonymous", p.Requests[0].Client)
	}
}

func TestAuth_KeysIdentifyTheClient(t *testing.T) {
	srv, p := authServer(t, map[string]string{"openwebui": webKey, "renovate-review": reviewKey})
	if code := do(t, "POST", srv.URL+"/v1/chat/completions", chatBody, map[string]string{"Authorization": "Bearer " + webKey}); code != 200 {
		t.Fatalf("bearer: status %d", code)
	}
	if code := do(t, "POST", srv.URL+"/v1/chat/completions", chatBody, map[string]string{"x-api-key": reviewKey}); code != 200 {
		t.Fatalf("x-api-key: status %d", code)
	}
	if len(p.Requests) != 2 || p.Requests[0].Client != "openwebui" || p.Requests[1].Client != "renovate-review" {
		t.Errorf("clients: %+v", p.Requests)
	}
}

func TestAuth_Rejects(t *testing.T) {
	srv, p := authServer(t, map[string]string{"openwebui": webKey})
	for name, h := range map[string]map[string]string{
		"missing":       nil,
		"wrong":         {"Authorization": "Bearer " + reviewKey},
		"prefix":        {"Authorization": "Bearer " + webKey[:20]},
		"not bearer":    {"Authorization": "Basic " + webKey},
		"empty api key": {"x-api-key": ""},
	} {
		if code := do(t, "POST", srv.URL+"/v1/chat/completions", chatBody, h); code != 401 {
			t.Errorf("%s: status %d, want 401", name, code)
		}
		if code := do(t, "GET", srv.URL+"/v1/models", "", h); code != 401 {
			t.Errorf("%s: /v1/models status %d, want 401", name, code)
		}
	}
	if len(p.Requests) != 0 {
		t.Error("an unauthenticated request reached the provider")
	}
}

// Probes carry no key.
func TestAuth_HealthzOpen(t *testing.T) {
	srv, _ := authServer(t, map[string]string{"openwebui": webKey})
	if code := do(t, "GET", srv.URL+"/healthz", "", nil); code != 200 {
		t.Errorf("status %d", code)
	}
}
