package httpapi

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Anonymous is the client name when llmux runs without client keys.
const Anonymous = "anonymous"

// Option configures the handler.
type Option func(*Handler)

// WithClientKeys turns on client authentication: every /v1 request must carry
// one of these keys, as "Authorization: Bearer <key>" (OpenAI clients) or
// "x-api-key: <key>" (Anthropic SDKs) — wherever a client would put a
// provider key. The key's name identifies the caller in telemetry. An empty
// map leaves authentication off and every caller is "anonymous".
func WithClientKeys(keys map[string]string) Option {
	return func(h *Handler) {
		for name, key := range keys {
			h.keys = append(h.keys, clientKey{name: name, sum: sha256.Sum256([]byte(key))})
		}
	}
}

// WithAuthFailureHook reports each rejected request's reason ("missing" or
// "invalid") — for telemetry, since the caller is unknown.
func WithAuthFailureHook(f func(reason string)) Option {
	return func(h *Handler) { h.onAuthFailure = f }
}

type clientKey struct {
	name string
	sum  [32]byte
}

type clientCtxKey struct{}

// clientFrom returns the authenticated client's name.
func clientFrom(ctx context.Context) string {
	if c, ok := ctx.Value(clientCtxKey{}).(string); ok {
		return c
	}
	return Anonymous
}

// authenticate wraps next so it only runs for a known client key. A request
// may carry the key in either header; it is accepted if either matches. Keys
// are compared as SHA-256 digests in constant time against every configured
// key, so timing reveals neither which key nor how much of one matched.
func (h *Handler) authenticate(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if len(h.keys) == 0 {
			next(w, r)
			return
		}
		var tokens []string
		if t := strings.TrimSpace(r.Header.Get("x-api-key")); t != "" {
			tokens = append(tokens, t)
		}
		if t, ok := bearerToken(r.Header.Get("Authorization")); ok {
			tokens = append(tokens, t)
		}
		name := ""
		for _, t := range tokens {
			sum := sha256.Sum256([]byte(t))
			for _, k := range h.keys {
				if subtle.ConstantTimeCompare(sum[:], k.sum[:]) == 1 {
					name = k.name
				}
			}
		}
		if name == "" {
			reason := "invalid"
			if len(tokens) == 0 {
				reason = "missing"
			}
			h.authFailed(r, reason)
			w.Header().Set("WWW-Authenticate", `Bearer realm="llmux"`)
			writeError(w, http.StatusUnauthorized, "missing or invalid llmux client key")
			return
		}
		next(w, r.WithContext(context.WithValue(r.Context(), clientCtxKey{}, name)))
	}
}

// bearerToken extracts the credentials of an "Authorization: Bearer" header.
// The scheme is case-insensitive (RFC 7235) and may be followed by spaces or
// a tab.
func bearerToken(header string) (string, bool) {
	scheme, rest, ok := strings.Cut(strings.TrimSpace(header), " ")
	if !ok {
		scheme, rest, ok = strings.Cut(strings.TrimSpace(header), "\t")
	}
	if !ok || !strings.EqualFold(scheme, "Bearer") {
		return "", false
	}
	t := strings.TrimSpace(rest)
	return t, t != ""
}

// authFailed counts a rejection and logs it, at most once per interval with a
// count, so a misconfigured client — or someone probing — is visible without
// flooding the log. The token is never logged.
func (h *Handler) authFailed(r *http.Request, reason string) {
	if h.onAuthFailure != nil {
		h.onAuthFailure(reason)
	}
	h.authLog.Lock()
	defer h.authLog.Unlock()
	h.authLog.suppressed++
	if time.Since(h.authLog.last) < authLogInterval {
		return
	}
	slog.Warn("rejected request: missing or invalid llmux client key",
		"reason", reason, "remote", r.RemoteAddr, "path", r.URL.Path, "rejections_since_last_log", h.authLog.suppressed)
	h.authLog.last, h.authLog.suppressed = time.Now(), 0
}

const authLogInterval = 10 * time.Second

type authLog struct {
	sync.Mutex
	last       time.Time
	suppressed int
}
