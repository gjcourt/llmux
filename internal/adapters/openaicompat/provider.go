// Package openaicompat is the outbound adapter for OpenAI-compatible model
// servers — vLLM and Ollama. It keeps llmux's original behaviour: requests with
// tools go to vLLM, falling back to Ollama through the tool-call repair
// transform; plain chat goes to Ollama, falling back to vLLM.
package openaicompat

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gjcourt/llmux/internal/domain"
	"github.com/gjcourt/llmux/internal/ports/outbound"
)

// Config selects the upstreams. An empty URL disables that upstream.
type Config struct {
	VLLMURL   string // e.g. http://host:8000 (the adapter appends /v1/...)
	OllamaURL string // e.g. http://host:11434/v1 (the adapter appends /...)
	Client    *http.Client
}

// Provider implements outbound.ChatProvider for vLLM and Ollama.
type Provider struct {
	cfg Config
}

var _ outbound.ChatProvider = (*Provider)(nil)

// New returns a Provider. A nil Client means http.DefaultClient.
func New(cfg Config) *Provider {
	if cfg.Client == nil {
		cfg.Client = http.DefaultClient
	}
	return &Provider{cfg: cfg}
}

// Name implements outbound.ChatProvider.
func (p *Provider) Name() string { return "openaicompat" }

// Enabled reports whether at least one upstream is configured.
func (p *Provider) Enabled() bool { return p.cfg.VLLMURL != "" || p.cfg.OllamaURL != "" }

// Handles implements outbound.ChatProvider. This provider is the catch-all: it
// serves any model id, because vLLM and Ollama each decide what they host.
func (p *Provider) Handles(string) bool { return p.Enabled() }

func (p *Provider) vllmChat() string {
	if p.cfg.VLLMURL == "" {
		return ""
	}
	return p.cfg.VLLMURL + "/v1/chat/completions"
}

func (p *Provider) ollamaChat() string {
	if p.cfg.OllamaURL == "" {
		return ""
	}
	return p.cfg.OllamaURL + "/chat/completions"
}

// Chat implements outbound.ChatProvider.
func (p *Provider) Chat(ctx context.Context, req domain.ChatRequest, sink domain.EventSink) error {
	body := req.Raw
	if hasTools(body) {
		// vLLM handles tools natively; fall back to Ollama with the
		// <tool_call> → tool_calls transform.
		if ok, err := p.forward(ctx, p.vllmChat(), body, sink); ok {
			return err
		}
		slog.Warn("vLLM unavailable, falling back to ollama with tool transform")
		if p.ollamaChat() == "" {
			return domain.ErrUnavailable
		}
		return p.transform(ctx, p.ollamaChat(), forceNoStream(body), body, false, sink)
	}
	if ok, err := p.forward(ctx, p.ollamaChat(), body, sink); ok {
		return err
	}
	if ok, err := p.forward(ctx, p.vllmChat(), body, sink); ok {
		return err
	}
	return domain.ErrUnavailable
}

// post sends body to target. It returns (nil, nil) when target is empty.
func (p *Provider) post(ctx context.Context, target string, body []byte) (*http.Response, error) {
	if target == "" {
		return nil, nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return p.cfg.Client.Do(req)
}

// forward relays target's answer (streamed or not) as events. ok is false only
// when target is disabled or unreachable, so the caller can fall back.
func (p *Provider) forward(ctx context.Context, target string, body []byte, sink domain.EventSink) (ok bool, err error) {
	resp, err := p.post(ctx, target, body)
	if err != nil {
		slog.Error("upstream unreachable", "target", target, "err", err)
		return false, nil
	}
	if resp == nil {
		return false, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		return true, upstreamError(resp)
	}
	if strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		return true, relaySSE(resp.Body, sink)
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return true, fmt.Errorf("read upstream response: %w", err)
	}
	var parsed openAIResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return true, fmt.Errorf("decode upstream response: %w", err)
	}
	return true, emitMessage(parsed, sink)
}

// transform buffers target's answer, repairs it with applyToolCallTransform,
// and emits the result. originalBody is the pre-forceNoStream request, used to
// retry as plain chat when the model returns nothing.
func (p *Provider) transform(ctx context.Context, target string, body, originalBody []byte, isRetry bool, sink domain.EventSink) error {
	resp, err := p.post(ctx, target, body)
	if err != nil {
		slog.Error("upstream unreachable", "target", target, "err", err)
		return domain.ErrUnavailable
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		return upstreamError(resp)
	}
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read upstream response: %w", err)
	}

	transformed, err := applyToolCallTransform(respBody)
	if err != nil {
		slog.Warn("tool call transform failed, passing through", "err", err)
		transformed = respBody
	}
	slog.Debug("transform result", "raw", string(respBody[:min(len(respBody), 500)]), "transformed", string(transformed[:min(len(transformed), 500)]))

	// Ollama sometimes returns empty content when tools are present but not
	// needed. Retry without tools, through this same transform path so any
	// <tool_call> XML in the retry is still extracted. isRetry stops a second
	// empty answer from recursing. Don't retry once the history holds tool
	// results: stripping tools from such a history makes the model hallucinate
	// partial <tool_call> fragments.
	if isEmptyNonToolResponse(transformed) && !isRetry && !hasToolResultMessages(originalBody) {
		slog.Warn("ollama returned empty response with tools, retrying as plain chat")
		stripped := stripTools(originalBody)
		return p.transform(ctx, target, forceNoStream(stripped), stripped, true, sink)
	}

	var parsed openAIResponse
	if err := json.Unmarshal(transformed, &parsed); err != nil {
		return fmt.Errorf("decode upstream response: %w", err)
	}
	return emitMessage(parsed, sink)
}

func upstreamError(resp *http.Response) error {
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	return &domain.UpstreamError{Status: resp.StatusCode, ContentType: resp.Header.Get("Content-Type"), Body: b}
}

// emitMessage replays a complete (non-streamed) response as events, in the
// same order the original writeAsSSE produced SSE chunks: role, then either the
// tool calls (name, then arguments, per call) or the text, then the finish
// reason.
func emitMessage(resp openAIResponse, sink domain.EventSink) error {
	if len(resp.Choices) == 0 {
		return nil
	}
	c := resp.Choices[0]
	events := []domain.Event{{Kind: domain.EventStart, ID: resp.ID, Model: resp.Model, Created: resp.Created}}

	if len(c.Message.ToolCalls) > 0 {
		for i, tc := range c.Message.ToolCalls {
			events = append(events,
				domain.Event{Kind: domain.EventToolCall, ToolCall: domain.ToolCallDelta{Index: i, ID: tc.ID, Type: "function", Name: tc.Function.Name}},
				domain.Event{Kind: domain.EventToolCall, ToolCall: domain.ToolCallDelta{Index: i, ArgumentsDelta: tc.Function.Arguments}},
			)
		}
		events = append(events, domain.Event{Kind: domain.EventFinish, FinishReason: "tool_calls"})
	} else {
		if c.Message.Content != nil && *c.Message.Content != "" {
			events = append(events, domain.Event{Kind: domain.EventText, Text: *c.Message.Content})
		}
		fr := c.FinishReason
		if fr == "" {
			fr = "stop"
		}
		events = append(events, domain.Event{Kind: domain.EventFinish, FinishReason: fr})
	}
	if u, ok := parseUsage(resp.Usage); ok {
		events = append(events, domain.Event{Kind: domain.EventUsage, Usage: u})
	}
	for _, e := range events {
		if err := sink.Emit(e); err != nil {
			return err
		}
	}
	return nil
}

// streamChunk is one OpenAI chat.completion.chunk.
type streamChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Created int64  `json:"created"`
	Choices []struct {
		Delta struct {
			Content   *string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
}

// relaySSE translates an upstream OpenAI SSE stream into events.
func relaySSE(r io.Reader, sink domain.EventSink) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 4<<20)
	started := false
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), "\r")
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			return nil
		}
		var ch streamChunk
		if err := json.Unmarshal([]byte(data), &ch); err != nil {
			slog.Warn("skipping unparseable upstream chunk", "err", err, "data", data[:min(len(data), 200)])
			continue
		}
		if !started {
			started = true
			if err := sink.Emit(domain.Event{Kind: domain.EventStart, ID: ch.ID, Model: ch.Model, Created: ch.Created}); err != nil {
				return err
			}
		}
		for _, c := range ch.Choices {
			if c.Delta.Content != nil && *c.Delta.Content != "" {
				if err := sink.Emit(domain.Event{Kind: domain.EventText, Text: *c.Delta.Content}); err != nil {
					return err
				}
			}
			for _, tc := range c.Delta.ToolCalls {
				d := domain.ToolCallDelta{Index: tc.Index, ID: tc.ID, Type: tc.Type, Name: tc.Function.Name, ArgumentsDelta: tc.Function.Arguments}
				if err := sink.Emit(domain.Event{Kind: domain.EventToolCall, ToolCall: d}); err != nil {
					return err
				}
			}
			if c.FinishReason != nil && *c.FinishReason != "" {
				if err := sink.Emit(domain.Event{Kind: domain.EventFinish, FinishReason: *c.FinishReason}); err != nil {
					return err
				}
			}
		}
		if u, ok := parseUsage(ch.Usage); ok {
			if err := sink.Emit(domain.Event{Kind: domain.EventUsage, Usage: u}); err != nil {
				return err
			}
		}
	}
	if err := sc.Err(); err != nil && !errors.Is(err, context.Canceled) {
		return fmt.Errorf("read upstream stream: %w", err)
	}
	return nil
}

func parseUsage(raw json.RawMessage) (domain.Usage, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return domain.Usage{}, false
	}
	var u struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	}
	if err := json.Unmarshal(raw, &u); err != nil {
		return domain.Usage{}, false
	}
	return domain.Usage{PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens, TotalTokens: u.TotalTokens}, true
}

// Models implements outbound.ChatProvider by merging both upstreams' lists.
func (p *Provider) Models(ctx context.Context) ([]domain.Model, error) {
	var out []domain.Model
	if p.cfg.VLLMURL != "" {
		out = append(out, p.fetchModels(ctx, p.cfg.VLLMURL+"/v1/models")...)
	}
	if p.cfg.OllamaURL != "" {
		out = append(out, p.fetchModels(ctx, p.cfg.OllamaURL+"/models")...)
	}
	return out, nil
}

func (p *Provider) fetchModels(ctx context.Context, url string) []domain.Model {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		slog.Warn("failed to create models request", "url", url, "err", err)
		return nil
	}
	resp, err := p.cfg.Client.Do(req)
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
	out := make([]domain.Model, 0, len(result.Data))
	for _, raw := range result.Data {
		var m struct {
			ID string `json:"id"`
		}
		_ = json.Unmarshal(raw, &m)
		out = append(out, domain.Model{ID: m.ID, Raw: raw})
	}
	return out
}
