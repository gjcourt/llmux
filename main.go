package main

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"regexp"
	"strings"
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

func hasTools(body []byte) bool {
	var req struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	return len(req.Tools) > 0
}

// forceNoStream rewrites the request body with stream:false so we can buffer and transform the response.
func forceNoStream(body []byte) []byte {
	var req map[string]json.RawMessage
	if err := json.Unmarshal(body, &req); err != nil {
		return body
	}
	req["stream"] = json.RawMessage(`false`)
	result, _ := json.Marshal(req)
	return result
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusInternalServerError)
		return
	}

	if hasTools(body) {
		// vLLM handles tools natively; fall back to ollama with <tool_call> → tool_calls transform.
		slog.Info("routing to vllm (has tools)", "path", r.URL.Path)
		if proxyStream(w, r, vllmURL+"/v1/chat/completions", body) {
			return
		}
		slog.Warn("vLLM unavailable, falling back to ollama with tool transform")
		proxyTransform(w, r, ollamaURL+"/chat/completions", forceNoStream(body))
	} else {
		slog.Info("routing to ollama", "path", r.URL.Path)
		if proxyStream(w, r, ollamaURL+"/chat/completions", body) {
			return
		}
		proxyStream(w, r, vllmURL+"/v1/chat/completions", body)
	}
}

// proxyStream sends body to target with streaming support.
// Returns false only on a connection error so the caller can try a fallback.
func proxyStream(w http.ResponseWriter, r *http.Request, target string, body []byte) bool {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return false
	}
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
		slog.Error("upstream unreachable", "target", target, "err", err)
		return false
	}
	defer resp.Body.Close()

	for k, vs := range resp.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)

	flusher, canFlush := w.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				return true
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
			return true
		}
	}
	return true
}

// proxyTransform sends body to target, buffers the response, applies the tool-call transform, and writes the result.
func proxyTransform(w http.ResponseWriter, r *http.Request, target string, body []byte) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "failed to create upstream request", http.StatusInternalServerError)
		return
	}
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
		slog.Error("upstream unreachable", "target", target, "err", err)
		http.Error(w, "upstream request failed", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "failed to read upstream response", http.StatusBadGateway)
		return
	}

	transformed, err := applyToolCallTransform(respBody)
	if err != nil {
		slog.Warn("tool call transform failed, passing through", "err", err)
		transformed = respBody
	}

	for k, vs := range resp.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.Header().Del("Transfer-Encoding")
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(transformed)))
	w.WriteHeader(resp.StatusCode)
	w.Write(transformed) //nolint:errcheck
}

var (
	reThink    = regexp.MustCompile(`(?s)<think>.*?</think>\s*`)
	reToolCall = regexp.MustCompile(`(?s)<tool_call>\s*(.*?)\s*</tool_call>`)
)

type openAIResponse struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created int64           `json:"created"`
	Model   string          `json:"model"`
	Choices []openAIChoice  `json:"choices"`
	Usage   json.RawMessage `json:"usage,omitempty"`
}

type openAIChoice struct {
	Index        int        `json:"index"`
	Message      openAIMsg  `json:"message"`
	FinishReason string     `json:"finish_reason"`
}

type openAIMsg struct {
	Role      string           `json:"role"`
	Content   *string          `json:"content"`
	ToolCalls []openAIToolCall `json:"tool_calls,omitempty"`
}

type openAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openAIToolFunction `json:"function"`
}

type openAIToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// applyToolCallTransform parses <tool_call> XML out of content and converts it
// to the OpenAI tool_calls array format. Strips <think> blocks as well.
func applyToolCallTransform(body []byte) ([]byte, error) {
	var resp openAIResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return body, err
	}

	changed := false
	for i, choice := range resp.Choices {
		if choice.Message.Content == nil {
			continue
		}
		content := *choice.Message.Content

		// Strip thinking tokens.
		content = reThink.ReplaceAllString(content, "")

		matches := reToolCall.FindAllStringSubmatch(content, -1)
		if len(matches) == 0 {
			resp.Choices[i].Message.Content = strPtr(strings.TrimSpace(content))
			continue
		}

		var calls []openAIToolCall
		for j, m := range matches {
			var tc struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}
			if err := json.Unmarshal([]byte(m[1]), &tc); err != nil {
				slog.Warn("failed to parse tool_call JSON", "err", err)
				continue
			}
			argStr, _ := json.Marshal(tc.Arguments)
			calls = append(calls, openAIToolCall{
				ID:   randomCallID(j),
				Type: "function",
				Function: openAIToolFunction{
					Name:      tc.Name,
					Arguments: string(argStr),
				},
			})
		}

		remaining := strings.TrimSpace(reToolCall.ReplaceAllString(content, ""))
		if remaining == "" {
			resp.Choices[i].Message.Content = nil
		} else {
			resp.Choices[i].Message.Content = strPtr(remaining)
		}
		resp.Choices[i].Message.ToolCalls = calls
		resp.Choices[i].FinishReason = "tool_calls"
		changed = true
	}

	if !changed {
		return body, nil
	}
	return json.Marshal(resp)
}

func strPtr(s string) *string { return &s }

func randomCallID(idx int) string {
	b := make([]byte, 4)
	rand.Read(b) //nolint:errcheck
	return fmt.Sprintf("call_%s%d", hex.EncodeToString(b), idx)
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	vllmModels := fetchModels(vllmURL + "/v1/models")
	ollamaModels := fetchModels(ollamaURL + "/models")

	merged := map[string]any{
		"object": "list",
		"data":   append(vllmModels, ollamaModels...),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(merged) //nolint:errcheck
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
