package main

import (
	"bytes"
	"context"
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
	"time"
)

// SSE types for converting a buffered non-streaming response back to the
// streaming wire format that clients expecting stream:true can consume.
type sseChunk struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created,omitempty"`
	Model   string      `json:"model"`
	Choices []sseChoice `json:"choices"`
}

type sseChoice struct {
	Index        int      `json:"index"`
	Delta        sseDelta `json:"delta"`
	FinishReason *string  `json:"finish_reason"`
}

type sseDelta struct {
	Role      string             `json:"role,omitempty"`
	Content   *string            `json:"content,omitempty"`
	ToolCalls []sseToolCallDelta `json:"tool_calls,omitempty"`
}

type sseToolCallDelta struct {
	Index    int               `json:"index"`
	ID       string            `json:"id,omitempty"`
	Type     string            `json:"type,omitempty"`
	Function *sseToolFuncDelta `json:"function,omitempty"`
}

type sseToolFuncDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

var (
	addr      = getenv("LLMUX_ADDR", ":8080")
	vllmURL   = getenv("LLMUX_VLLM_URL", "http://10.42.2.10:8000")
	ollamaURL = getenv("LLMUX_OLLAMA_URL", "http://10.42.2.10:30068/v1")

	httpClient = &http.Client{Timeout: 120 * time.Second}
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

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))
	slog.Info("llmux listening", "addr", addr, "vllm", vllmURL, "ollama", ollamaURL)
	if err := http.ListenAndServe(addr, mux); err != nil {
		slog.Error("server error", "err", err)
		os.Exit(1)
	}
}

func requestedStream(body []byte) bool {
	var req struct {
		Stream bool `json:"stream"`
	}
	_ = json.Unmarshal(body, &req)
	return req.Stream
}

// writeAsSSE converts a buffered non-streaming OpenAI response into SSE chunks
// and writes them to w. Used when the original request had stream:true but we
// had to buffer the response to apply the tool-call transform.
func writeAsSSE(w http.ResponseWriter, body []byte) {
	flusher, canFlush := w.(http.Flusher)
	emit := func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if canFlush {
			flusher.Flush()
		}
	}

	var resp openAIResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		fmt.Fprint(w, "data: [DONE]\n\n")
		if canFlush {
			flusher.Flush()
		}
		return
	}

	base := sseChunk{ID: resp.ID, Object: "chat.completion.chunk", Created: resp.Created, Model: resp.Model}
	var nullFR *string

	if len(resp.Choices) == 0 {
		fmt.Fprint(w, "data: [DONE]\n\n")
		if canFlush {
			flusher.Flush()
		}
		return
	}
	c := resp.Choices[0]

	// Role delta — always first.
	chunk := base
	chunk.Choices = []sseChoice{{Index: c.Index, Delta: sseDelta{Role: "assistant"}, FinishReason: nullFR}}
	emit(chunk)

	if len(c.Message.ToolCalls) > 0 {
		for i, tc := range c.Message.ToolCalls {
			chunk = base
			chunk.Choices = []sseChoice{{
				Index: c.Index,
				Delta: sseDelta{ToolCalls: []sseToolCallDelta{{
					Index: i, ID: tc.ID, Type: "function",
					Function: &sseToolFuncDelta{Name: tc.Function.Name},
				}}},
				FinishReason: nullFR,
			}}
			emit(chunk)

			chunk = base
			chunk.Choices = []sseChoice{{
				Index: c.Index,
				Delta: sseDelta{ToolCalls: []sseToolCallDelta{{
					Index:    i,
					Function: &sseToolFuncDelta{Arguments: tc.Function.Arguments},
				}}},
				FinishReason: nullFR,
			}}
			emit(chunk)
		}
		fr := "tool_calls"
		chunk = base
		chunk.Choices = []sseChoice{{Index: c.Index, Delta: sseDelta{}, FinishReason: &fr}}
		emit(chunk)
	} else {
		if c.Message.Content != nil && *c.Message.Content != "" {
			content := *c.Message.Content
			chunk = base
			chunk.Choices = []sseChoice{{Index: c.Index, Delta: sseDelta{Content: &content}, FinishReason: nullFR}}
			emit(chunk)
		}
		fr := c.FinishReason
		if fr == "" {
			fr = "stop"
		}
		chunk = base
		chunk.Choices = []sseChoice{{Index: c.Index, Delta: sseDelta{}, FinishReason: &fr}}
		emit(chunk)
	}

	fmt.Fprint(w, "data: [DONE]\n\n")
	if canFlush {
		flusher.Flush()
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

// stripTools removes tools and tool_choice from the request body so the model responds as plain chat.
func stripTools(body []byte) []byte {
	var req map[string]json.RawMessage
	if err := json.Unmarshal(body, &req); err != nil {
		return body
	}
	delete(req, "tools")
	delete(req, "tool_choice")
	result, _ := json.Marshal(req)
	return result
}

// isEmptyNonToolResponse returns true when the model responded with no content and no tool calls —
// meaning the model silently gave up rather than answering or calling a tool.
func isEmptyNonToolResponse(body []byte) bool {
	var resp openAIResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return false
	}
	if len(resp.Choices) == 0 {
		return true
	}
	c := resp.Choices[0]
	if c.FinishReason == "tool_calls" || len(c.Message.ToolCalls) > 0 {
		return false
	}
	return c.Message.Content == nil || *c.Message.Content == ""
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusInternalServerError)
		return
	}

	slog.Debug("incoming request", "body", string(body[:min(len(body), 300)]))

	if hasTools(body) {
		// vLLM handles tools natively; fall back to ollama with <tool_call> → tool_calls transform.
		slog.Info("routing to vllm (has tools)", "path", r.URL.Path)
		if proxyStream(w, r, vllmURL+"/v1/chat/completions", body) {
			return
		}
		slog.Warn("vLLM unavailable, falling back to ollama with tool transform")
		proxyTransform(w, r, ollamaURL+"/chat/completions", forceNoStream(body), body)
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

	resp, err := httpClient.Do(req)
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
// originalBody is the pre-forceNoStream request body; it's used to retry as plain chat if the model returns empty.
func proxyTransform(w http.ResponseWriter, r *http.Request, target string, body []byte, originalBody []byte) {
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

	resp, err := httpClient.Do(req)
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
	slog.Debug("transform result", "raw", string(respBody[:min(len(respBody), 500)]), "transformed", string(transformed[:min(len(transformed), 500)]))

	// Ollama sometimes returns empty content when tools are present but not needed.
	// Retry as plain chat (no tools) so the model can respond conversationally.
	if isEmptyNonToolResponse(transformed) {
		slog.Warn("ollama returned empty response with tools, retrying as plain chat")
		proxyStream(w, r, target, stripTools(originalBody))
		return
	}

	if requestedStream(originalBody) {
		slog.Debug("converting buffered response to SSE")
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(resp.StatusCode)
		writeAsSSE(w, transformed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	w.Write(transformed) //nolint:errcheck
}

var (
	// Ollama sometimes strips the <think>...</think> content but leaves the closing tag.
	reThink    = regexp.MustCompile(`(?s)(?:<think>.*?</think>|</think>)\s*`)
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
	Index        int       `json:"index"`
	Message      openAIMsg `json:"message"`
	FinishReason string    `json:"finish_reason"`
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
		stripped := reThink.ReplaceAllString(content, "")

		matches := reToolCall.FindAllStringSubmatch(stripped, -1)
		if len(matches) == 0 {
			if stripped != content {
				// Think tags were present; update the choice even though no tool calls found.
				resp.Choices[i].Message.Content = strPtr(strings.TrimSpace(stripped))
				changed = true
			}
			continue
		}

		var calls []openAIToolCall
		for j, m := range matches {
			raw := m[1]
			var tc struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}
			if err := json.Unmarshal([]byte(raw), &tc); err != nil {
				// Models sometimes emit literal newlines inside JSON strings (invalid JSON).
				// Repair by escaping control characters within string values and retry.
				if repaired := repairJSONStrings(raw); repaired != raw {
					if err2 := json.Unmarshal([]byte(repaired), &tc); err2 != nil {
						slog.Warn("failed to parse tool_call JSON after repair", "err", err2)
						continue
					}
				} else {
					slog.Warn("failed to parse tool_call JSON", "err", err)
					continue
				}
			}
			calls = append(calls, openAIToolCall{
				ID:   randomCallID(j),
				Type: "function",
				Function: openAIToolFunction{
					Name:      tc.Name,
					Arguments: string(tc.Arguments),
				},
			})
		}

		// If all tool call parses failed, leave this choice unchanged.
		if len(calls) == 0 {
			resp.Choices[i].Message.Content = strPtr(strings.TrimSpace(content))
			continue
		}

		// OpenAI spec: content must be null when tool_calls are present.
		// Some clients (hermes) reject responses that have both.
		resp.Choices[i].Message.Content = nil
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
	rand.Read(b) //nolint:errcheck // crypto/rand.Read never returns an error since Go 1.20
	return fmt.Sprintf("call_%s%d", hex.EncodeToString(b), idx)
}

// repairJSONStrings escapes bare control characters (newlines, tabs, carriage returns)
// that appear inside JSON string values. Models often emit these literally in code arguments.
func repairJSONStrings(s string) string {
	var b strings.Builder
	inString := false
	escaped := false
	b.Grow(len(s))
	for _, c := range s {
		if escaped {
			escaped = false
			b.WriteRune(c)
			continue
		}
		if c == '\\' {
			escaped = true
			b.WriteRune(c)
			continue
		}
		if c == '"' {
			inString = !inString
			b.WriteRune(c)
			continue
		}
		if inString {
			switch c {
			case '\n':
				b.WriteString(`\n`)
			case '\r':
				b.WriteString(`\r`)
			case '\t':
				b.WriteString(`\t`)
			default:
				b.WriteRune(c)
			}
		} else {
			b.WriteRune(c)
		}
	}
	return b.String()
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	vllmModels := fetchModels(ctx, vllmURL+"/v1/models")
	ollamaModels := fetchModels(ctx, ollamaURL+"/models")

	merged := map[string]any{
		"object": "list",
		"data":   append(vllmModels, ollamaModels...),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(merged) //nolint:errcheck
}

func fetchModels(ctx context.Context, url string) []json.RawMessage {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		slog.Warn("failed to create models request", "url", url, "err", err)
		return nil
	}
	resp, err := httpClient.Do(req)
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
