// Package httpapi is the inbound adapter: an OpenAI-compatible HTTP API over
// inbound.ChatService.
package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"

	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/ports/inbound"
)

// maxBody caps a request body. Open WebUI sends images inline as base64, so
// this must be well above a chat history's size; the original proxy had no
// limit at all.
const maxBody = 64 << 20

// Handler serves the OpenAI-compatible routes.
type Handler struct {
	svc  inbound.ChatService
	keys []clientKey
}

// New returns the HTTP handler for svc.
func New(svc inbound.ChatService, opts ...Option) http.Handler {
	h := &Handler{svc: svc}
	for _, o := range opts {
		o(h)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", h.authenticate(h.chat))
	mux.HandleFunc("GET /v1/models", h.authenticate(h.models))
	mux.HandleFunc("GET /healthz", h.healthz) // unauthenticated: probes
	return mux
}

func (h *Handler) healthz(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	w.Write([]byte("ok\n")) //nolint:errcheck
}

func (h *Handler) chat(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body exceeds 64 MiB")
			return
		}
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	req, err := parseRequest(body)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	req.Client = clientFrom(r.Context())
	slog.Debug("incoming request", "model", req.Model, "stream", req.Stream, "len", len(body))

	if req.Stream {
		sink := newSSESink(w, req.IncludeUsage)
		err := h.svc.Chat(r.Context(), req, sink)
		switch {
		case err == nil:
			sink.done()
		case !sink.started:
			writeChatError(w, err)
		case errors.Is(err, context.Canceled):
			// Client went away mid-stream; nothing left to tell it.
		default:
			slog.Error("chat failed mid-stream", "err", err)
			sink.fail(err.Error())
		}
		return
	}

	sink := &jsonSink{}
	if err := h.svc.Chat(r.Context(), req, sink); err != nil {
		writeChatError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(sink.response()) //nolint:errcheck
}

// writeChatError maps a service error to an HTTP response. An upstream's own
// error response is relayed with its body and, mostly, its status; see
// clientStatus.
func writeChatError(w http.ResponseWriter, err error) {
	var ue *domain.UpstreamError
	var ie *domain.InvalidRequestError
	switch {
	case errors.As(err, &ie):
		writeError(w, http.StatusBadRequest, ie.Msg)
	case errors.As(err, &ue):
		ct := ue.ContentType
		if ct == "" {
			ct = "application/json"
		}
		w.Header().Set("Content-Type", ct)
		if ue.RetryAfter != "" {
			w.Header().Set("Retry-After", ue.RetryAfter)
		}
		w.WriteHeader(clientStatus(ue.Status))
		w.Write(ue.Body) //nolint:errcheck
	case errors.Is(err, domain.ErrNoProvider):
		writeError(w, http.StatusNotFound, err.Error())
	case errors.Is(err, domain.ErrUnavailable):
		writeError(w, http.StatusBadGateway, err.Error())
	case errors.Is(err, context.Canceled):
		// Client went away; nobody to answer.
	default:
		slog.Error("chat failed", "err", err)
		writeError(w, http.StatusBadGateway, err.Error())
	}
}

// clientStatus is the status a client sees for an upstream's. Redirects and
// auth failures are llmux's own configuration, not the client's, so they
// become 502 — an auth failure must not tell the client to re-authenticate.
// Anthropic's non-standard 529 (overloaded) becomes 503, which clients know
// to retry.
func clientStatus(upstream int) int {
	if upstream >= 300 && upstream < 400 {
		// A 3xx reaches here only when a provider didn't follow it (the
		// Anthropic client never does; see anthropic.HTTPClient). A bare 3xx
		// without its Location means nothing to the client.
		slog.Warn("upstream redirected; check the backend base URL", "status", upstream)
		return http.StatusBadGateway
	}
	switch upstream {
	case http.StatusUnauthorized, http.StatusForbidden:
		return http.StatusBadGateway
	case 529:
		return http.StatusServiceUnavailable
	}
	return upstream
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)   // this is an API, not HTML: keep "n > 1" readable
	enc.Encode(map[string]any{ //nolint:errcheck
		"error": map[string]string{"message": msg, "type": http.StatusText(status)},
	})
}

func (h *Handler) models(w http.ResponseWriter, r *http.Request) {
	ms, err := h.svc.Models(r.Context())
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	data := make([]json.RawMessage, 0, len(ms))
	for _, m := range ms {
		if len(m.Raw) > 0 {
			data = append(data, m.Raw)
			continue
		}
		b, _ := json.Marshal(map[string]any{"id": m.ID, "object": "model", "owned_by": m.OwnedBy})
		data = append(data, b)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data}) //nolint:errcheck
}
