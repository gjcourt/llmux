package httpapi

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"net/http"
	"strings"
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

// authenticate wraps next so it only runs for a known client key. Keys are
// compared as SHA-256 digests in constant time, and every configured key is
// checked, so timing reveals neither which key nor how much of one matched.
func (h *Handler) authenticate(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if len(h.keys) == 0 {
			next(w, r)
			return
		}
		token := r.Header.Get("x-api-key")
		if bearer, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer "); ok && token == "" {
			token = strings.TrimSpace(bearer)
		}
		sum := sha256.Sum256([]byte(token))
		name := ""
		for _, k := range h.keys {
			if subtle.ConstantTimeCompare(sum[:], k.sum[:]) == 1 {
				name = k.name
			}
		}
		if token == "" || name == "" {
			writeError(w, http.StatusUnauthorized, "missing or invalid llmux client key")
			return
		}
		next(w, r.WithContext(context.WithValue(r.Context(), clientCtxKey{}, name)))
	}
}
