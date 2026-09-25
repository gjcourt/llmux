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

// maxBody caps a request body. Chat histories are text; 16 MiB is generous.
const maxBody = 16 << 20

// Handler serves the OpenAI-compatible routes.
type Handler struct {
	svc inbound.ChatService
}

// New returns the HTTP handler for svc.
func New(svc inbound.ChatService) http.Handler {
	h := &Handler{svc: svc}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", h.chat)
	mux.HandleFunc("GET /v1/models", h.models)
	mux.HandleFunc("GET /healthz", h.healthz)
	return mux
}

func (h *Handler) healthz(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	w.Write([]byte("ok\n")) //nolint:errcheck
}

func (h *Handler) chat(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBody))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	req, err := parseRequest(body)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
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
// error response is relayed with its original status and body.
func writeChatError(w http.ResponseWriter, err error) {
	var ue *domain.UpstreamError
	switch {
	case errors.As(err, &ue):
		ct := ue.ContentType
		if ct == "" {
			ct = "application/json"
		}
		w.Header().Set("Content-Type", ct)
		w.WriteHeader(ue.Status)
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

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
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
