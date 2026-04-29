package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"os"
)

var (
	addr      = getenv("LLMUX_ADDR", ":8080")
	vllmURL   = getenv("LLMUX_VLLM_URL", "http://10.42.2.10:8000")
	ollamaURL = getenv("LLMUX_OLLAMA_URL", "http://10.42.2.10:30068/v1")
)

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", handleChat)
	mux.HandleFunc("GET /v1/models", handleModels)

	slog.Info("llmux listening", "addr", addr, "vllm", vllmURL, "ollama", ollamaURL)
	if err := http.ListenAndServe(addr, mux); err != nil {
		slog.Error("server error", "err", err)
		os.Exit(1)
	}
}

// hasTools returns true if the request body contains a non-empty "tools" array.
func hasTools(body []byte) bool {
	var req struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	return len(req.Tools) > 0
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusInternalServerError)
		return
	}

	var target string
	if hasTools(body) {
		target = vllmURL + "/v1/chat/completions"
		slog.Info("routing to vllm", "path", r.URL.Path)
	} else {
		target = ollamaURL + "/chat/completions"
		slog.Info("routing to ollama", "path", r.URL.Path)
	}

	proxy(w, r, target, body)
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	vllmModels := fetchModels(vllmURL + "/v1/models")
	ollamaModels := fetchModels(ollamaURL + "/models")

	merged := map[string]any{
		"object": "list",
		"data":   append(vllmModels, ollamaModels...),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(merged)
}

func fetchModels(url string) []json.RawMessage {
	resp, err := http.Get(url)
	if err != nil {
		slog.Warn("failed to fetch models", "url", url, "err", err)
		return nil
	}
	defer resp.Body.Close()

	var result struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		slog.Warn("failed to decode models response", "url", url, "err", err)
		return nil
	}
	return result.Data
}

func proxy(w http.ResponseWriter, r *http.Request, target string, body []byte) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "failed to create upstream request", http.StatusInternalServerError)
		return
	}

	// Forward relevant headers.
	for _, h := range []string{"Content-Type", "Authorization", "Accept"} {
		if v := r.Header.Get(h); v != "" {
			req.Header.Set(h, v)
		}
	}
	if req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		slog.Error("upstream request failed", "target", target, "err", err)
		http.Error(w, "upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Copy response headers.
	for k, vs := range resp.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)

	// Stream the response body, flushing as data arrives.
	flusher, canFlush := w.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				return
			}
			if canFlush {
				flusher.Flush()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			slog.Error("error reading upstream response", "err", readErr)
			return
		}
	}
}
