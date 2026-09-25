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

var failures []string

func authServer(t *testing.T, keys map[string]string) (*httptest.Server, *testdoubles.Provider) {
	t.Helper()
	failures = nil
	p := &testdoubles.Provider{Events: []domain.Event{{Kind: domain.EventStart}, {Kind: domain.EventText, Text: "hi"}, {Kind: domain.EventFinish, FinishReason: "stop"}}}
	srv := httptest.NewServer(httpapi.New(app.New(p), httpapi.WithClientKeys(keys),
		httpapi.WithAuthFailureHook(func(r string) { failures = append(failures, r) })))
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

// A request is accepted if either header carries a valid key (the Anthropic
// SDK can send both), and "Bearer" is case-insensitive.
func TestAuth_HeaderForms(t *testing.T) {
	srv, _ := authServer(t, map[string]string{"openwebui": webKey})
	for name, h := range map[string]map[string]string{
		"wrong x-api-key, valid bearer": {"x-api-key": "nope", "Authorization": "Bearer " + webKey},
		"valid x-api-key, wrong bearer": {"x-api-key": webKey, "Authorization": "Bearer nope"},
		"lowercase scheme":              {"Authorization": "bearer " + webKey},
		"tab after scheme":              {"Authorization": "Bearer\t" + webKey},
		"extra spaces":                  {"Authorization": "Bearer   " + webKey + "  "},
	} {
		if code := do(t, "POST", srv.URL+"/v1/chat/completions", chatBody, h); code != 200 {
			t.Errorf("%s: status %d, want 200", name, code)
		}
	}
}

// Rejections are counted by reason and carry a WWW-Authenticate challenge.
func TestAuth_FailuresReported(t *testing.T) {
	srv, _ := authServer(t, map[string]string{"openwebui": webKey})
	do(t, "POST", srv.URL+"/v1/chat/completions", chatBody, nil)
	do(t, "GET", srv.URL+"/v1/models", "", map[string]string{"Authorization": "Bearer " + reviewKey})
	if len(failures) != 2 || failures[0] != "missing" || failures[1] != "invalid" {
		t.Errorf("failures: %v", failures)
	}
	req, _ := http.NewRequest("GET", srv.URL+"/v1/models", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.Header.Get("WWW-Authenticate") == "" {
		t.Error("401 must carry WWW-Authenticate")
	}
}
