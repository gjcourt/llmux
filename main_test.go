package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// --- hasTools ---

func TestHasTools(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		body string
		want bool
	}{
		{"with tools", `{"model":"x","tools":[{"type":"function","function":{"name":"f"}}]}`, true},
		{"empty tools array", `{"model":"x","tools":[]}`, false},
		{"no tools key", `{"model":"x","messages":[]}`, false},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := hasTools([]byte(tt.body)); got != tt.want {
				t.Errorf("hasTools = %v, want %v", got, tt.want)
			}
		})
	}
}

// --- forceNoStream ---

func TestForceNoStream_AddsField(t *testing.T) {
	t.Parallel()
	body := `{"model":"x"}`
	out := forceNoStream([]byte(body))
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if string(m["stream"]) != "false" {
		t.Fatalf("expected stream=false, got %s", m["stream"])
	}
}

func TestForceNoStream_OverridesTrue(t *testing.T) {
	t.Parallel()
	body := `{"model":"x","stream":true}`
	out := forceNoStream([]byte(body))
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if string(m["stream"]) != "false" {
		t.Fatalf("expected stream=false, got %s", m["stream"])
	}
}

// --- applyToolCallTransform ---

func ollamaResp(content string) []byte {
	r := map[string]any{
		"id":     "chatcmpl-1",
		"object": "chat.completion",
		"model":  "qwen3.6-27b:latest",
		"choices": []map[string]any{{
			"index": 0,
			"message": map[string]any{
				"role":    "assistant",
				"content": content,
			},
			"finish_reason": "stop",
		}},
	}
	b, _ := json.Marshal(r)
	return b
}

type transformResult struct {
	FinishReason string
	Content      *string
	ToolCalls    []openAIToolCall
}

func parseTransform(t *testing.T, body []byte) transformResult {
	t.Helper()
	var resp openAIResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("failed to unmarshal transform output: %v\nbody: %s", err, body)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("no choices in transform output")
	}
	c := resp.Choices[0]
	return transformResult{
		FinishReason: c.FinishReason,
		Content:      c.Message.Content,
		ToolCalls:    c.Message.ToolCalls,
	}
}

func TestTransform_SingleToolCall(t *testing.T) {
	t.Parallel()
	input := ollamaResp(`<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>`)
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)

	if r.FinishReason != "tool_calls" {
		t.Errorf("finish_reason: want tool_calls, got %s", r.FinishReason)
	}
	if r.Content != nil {
		t.Errorf("content: want nil, got %q", *r.Content)
	}
	if len(r.ToolCalls) != 1 {
		t.Fatalf("tool_calls count: want 1, got %d", len(r.ToolCalls))
	}
	tc := r.ToolCalls[0]
	if tc.Function.Name != "web_search" {
		t.Errorf("tool name: want web_search, got %s", tc.Function.Name)
	}
	if tc.Type != "function" {
		t.Errorf("tool type: want function, got %s", tc.Type)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatalf("arguments not valid JSON: %v", err)
	}
	if args["query"] != "cats" {
		t.Errorf("argument query: want cats, got %s", args["query"])
	}
}

func TestTransform_StripCloseThinkTag(t *testing.T) {
	t.Parallel()
	// Ollama strips <think>...</think> content but leaves the </think> closing tag.
	input := ollamaResp("</think>\n\n<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)

	if r.Content != nil {
		t.Errorf("want nil content after stripping </think>, got %q", *r.Content)
	}
	if len(r.ToolCalls) != 1 {
		t.Errorf("want 1 tool call, got %d", len(r.ToolCalls))
	}
}

func TestTransform_StripFullThinkBlock(t *testing.T) {
	t.Parallel()
	input := ollamaResp("<think>reasoning here</think>\n<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)

	if r.Content != nil {
		t.Errorf("want nil content, got %q", *r.Content)
	}
	if len(r.ToolCalls) != 1 {
		t.Errorf("want 1 tool call, got %d", len(r.ToolCalls))
	}
}

// TestTransform_StripThinkNoToolCall verifies that <think> stripping works
// even when the model responds conversationally (no tool calls).
func TestTransform_StripThinkNoToolCall(t *testing.T) {
	t.Parallel()
	input := ollamaResp("<think>internal reasoning</think>\nHello! How can I help?")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)

	if r.Content == nil || *r.Content != "Hello! How can I help?" {
		t.Errorf("want think-stripped content, got %v", r.Content)
	}
	if len(r.ToolCalls) != 0 {
		t.Errorf("want no tool calls, got %d", len(r.ToolCalls))
	}
	if r.FinishReason != "stop" {
		t.Errorf("finish_reason should remain stop, got %s", r.FinishReason)
	}
}

func TestTransform_StripOrphanCloseThinkNoToolCall(t *testing.T) {
	t.Parallel()
	input := ollamaResp("</think>Hello!")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)
	if r.Content == nil || *r.Content != "Hello!" {
		t.Errorf("want orphan </think> stripped, got %v", r.Content)
	}
}

func TestTransform_MultipleToolCalls(t *testing.T) {
	t.Parallel()
	input := ollamaResp(
		"<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\n" +
			"<tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>",
	)
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)

	if len(r.ToolCalls) != 2 {
		t.Fatalf("want 2 tool calls, got %d", len(r.ToolCalls))
	}
	if r.ToolCalls[0].Function.Name != "a" || r.ToolCalls[1].Function.Name != "b" {
		t.Errorf("unexpected tool names: %v", r.ToolCalls)
	}
}

func TestTransform_ContentWithPreamble(t *testing.T) {
	t.Parallel()
	// Model says something before calling a tool.
	// Per OpenAI spec, content must be null when tool_calls are present.
	input := ollamaResp("Let me look that up.\n<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)

	if r.Content != nil {
		t.Errorf("content must be null when tool_calls present, got %q", *r.Content)
	}
	if len(r.ToolCalls) != 1 {
		t.Errorf("want 1 tool call, got %d", len(r.ToolCalls))
	}
}

func TestTransform_NoToolCall_PassThrough(t *testing.T) {
	t.Parallel()
	input := ollamaResp("Hello! How can I help you today?")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	// Output should be the same bytes (unchanged path).
	if string(out) != string(input) {
		t.Errorf("expected pass-through, but body changed:\nwant: %s\ngot:  %s", input, out)
	}
}

func TestTransform_EmptyContent_PassThrough(t *testing.T) {
	t.Parallel()
	input := ollamaResp("")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)
	if r.FinishReason != "stop" {
		t.Errorf("finish_reason should remain stop, got %s", r.FinishReason)
	}
	if len(r.ToolCalls) != 0 {
		t.Errorf("want no tool calls, got %d", len(r.ToolCalls))
	}
}

func TestTransform_MalformedToolCallJSON(t *testing.T) {
	t.Parallel()
	// Malformed JSON inside <tool_call> — should not panic, tool is skipped.
	input := ollamaResp("<tool_call>not json</tool_call>")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)
	if len(r.ToolCalls) != 0 {
		t.Errorf("malformed tool call should be skipped, got %d calls", len(r.ToolCalls))
	}
}

func TestTransform_InvalidTopLevelJSON(t *testing.T) {
	t.Parallel()
	_, err := applyToolCallTransform([]byte("not json"))
	if err == nil {
		t.Fatal("expected error for invalid JSON input")
	}
}

func TestTransform_TrailingGarbageAfterJSON(t *testing.T) {
	// Model sometimes emits valid JSON followed by a comment or Python continuation
	// outside the closing brace. json.Unmarshal rejects this; json.Decoder accepts it.
	input := ollamaResp("<tool_call>{\"name\":\"web_search\",\"arguments\":{\"query\":\"cats\"}}\n# trailing comment</tool_call>")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)
	if len(r.ToolCalls) != 1 {
		t.Fatalf("want 1 tool call despite trailing garbage, got %d", len(r.ToolCalls))
	}
	if r.ToolCalls[0].Function.Name != "web_search" {
		t.Errorf("tool name: want web_search, got %s", r.ToolCalls[0].Function.Name)
	}
}

func TestTransform_LiteralNewlinesInCode(t *testing.T) {
	// Model generates Python code with literal (unescaped) newlines inside the JSON string —
	// invalid JSON that we must repair before parsing.
	rawContent := "<tool_call>\n{\"name\": \"execute_code\", \"arguments\": {\"code\": \"import os\nprint(os.getcwd())\n\"}}\n</tool_call>"
	input := ollamaResp(rawContent)
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)

	if r.FinishReason != "tool_calls" {
		t.Errorf("finish_reason: want tool_calls, got %s", r.FinishReason)
	}
	if len(r.ToolCalls) != 1 {
		t.Fatalf("want 1 tool call, got %d — literal newlines in JSON not repaired", len(r.ToolCalls))
	}
	if r.ToolCalls[0].Function.Name != "execute_code" {
		t.Errorf("tool name: want execute_code, got %s", r.ToolCalls[0].Function.Name)
	}
}

func TestTransform_AllToolCallsParseFail_FinishReasonNotOverridden(t *testing.T) {
	// If every <tool_call> fails to parse, finish_reason must stay "stop" —
	// never emit finish_reason:"tool_calls" with an empty tool_calls array.
	input := ollamaResp("<tool_call>COMPLETELY INVALID</tool_call>")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)
	if r.FinishReason == "tool_calls" {
		t.Error("finish_reason must not be tool_calls when no tool calls were successfully parsed")
	}
	if len(r.ToolCalls) != 0 {
		t.Errorf("want 0 tool calls, got %d", len(r.ToolCalls))
	}
}

// --- stripTools ---

func TestStripTools_RemovesTools(t *testing.T) {
	t.Parallel()
	body := `{"model":"x","tools":[{"type":"function"}],"tool_choice":"auto","messages":[]}`
	out := stripTools([]byte(body))
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["tools"]; ok {
		t.Error("tools key should be removed")
	}
	if _, ok := m["tool_choice"]; ok {
		t.Error("tool_choice key should be removed")
	}
	if _, ok := m["messages"]; !ok {
		t.Error("messages key should be preserved")
	}
}

// --- isEmptyNonToolResponse ---

func TestIsEmptyNonToolResponse(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		input []byte
		want  bool
	}{
		{"empty content", ollamaResp(""), true},
		{"plain text", ollamaResp("Hello!"), false},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := isEmptyNonToolResponse(tt.input); got != tt.want {
				t.Errorf("isEmptyNonToolResponse = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsEmptyNonToolResponse_WithToolCalls(t *testing.T) {
	t.Parallel()
	// A properly transformed response with tool_calls should never be "empty".
	input := ollamaResp(`<tool_call>{"name":"f","arguments":{}}</tool_call>`)
	transformed, _ := applyToolCallTransform(input)
	if isEmptyNonToolResponse(transformed) {
		t.Fatal("response with tool_calls should not be considered empty")
	}
}

// --- hasToolResultMessages ---

func TestHasToolResultMessages_WithToolRole(t *testing.T) {
	body := `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"result"}]}`
	if !hasToolResultMessages([]byte(body)) {
		t.Fatal("expected true when role:tool present")
	}
}

func TestHasToolResultMessages_NoToolRole(t *testing.T) {
	body := `{"messages":[{"role":"user","content":"hi"}]}`
	if hasToolResultMessages([]byte(body)) {
		t.Fatal("expected false when no role:tool")
	}
}

// --- requestedStream ---

func TestRequestedStream_True(t *testing.T) {
	if !requestedStream([]byte(`{"model":"x","stream":true}`)) {
		t.Fatal("expected requestedStream=true")
	}
}

func TestRequestedStream_False(t *testing.T) {
	if requestedStream([]byte(`{"model":"x","stream":false}`)) {
		t.Fatal("expected requestedStream=false")
	}
}

func TestRequestedStream_Absent(t *testing.T) {
	if requestedStream([]byte(`{"model":"x"}`)) {
		t.Fatal("expected requestedStream=false when field absent")
	}
}

// --- writeAsSSE ---

// parseSSEEvents splits an SSE body into the data payloads (strips "data: " prefix and blank lines).
func parseSSEEvents(body string) []string {
	var events []string
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.HasPrefix(line, "data: ") {
			events = append(events, strings.TrimPrefix(line, "data: "))
		}
	}
	return events
}

func TestWriteAsSSE_ToolCall(t *testing.T) {
	input := ollamaResp(`<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>`)
	transformed, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	writeAsSSE(rr, transformed)

	events := parseSSEEvents(rr.Body.String())
	if len(events) == 0 {
		t.Fatal("no SSE events emitted")
	}
	if events[len(events)-1] != "[DONE]" {
		t.Errorf("last event should be [DONE], got %q", events[len(events)-1])
	}

	// Collect tool_call deltas.
	var toolCallName, toolCallArgs string
	for _, ev := range events {
		if ev == "[DONE]" {
			continue
		}
		var chunk sseChunk
		if err := json.Unmarshal([]byte(ev), &chunk); err != nil {
			t.Fatalf("invalid SSE JSON: %v\nevent: %s", err, ev)
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		for _, tc := range chunk.Choices[0].Delta.ToolCalls {
			if tc.Function != nil {
				toolCallName += tc.Function.Name
				toolCallArgs += tc.Function.Arguments
			}
		}
	}
	if toolCallName != "web_search" {
		t.Errorf("tool name: want web_search, got %q", toolCallName)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(toolCallArgs), &args); err != nil {
		t.Fatalf("tool arguments not valid JSON: %v", err)
	}
	if args["query"] != "cats" {
		t.Errorf("tool arg query: want cats, got %q", args["query"])
	}

	// Final chunk must have finish_reason: tool_calls.
	var finalChunk sseChunk
	for _, ev := range events {
		if ev == "[DONE]" {
			continue
		}
		var ch sseChunk
		if err := json.Unmarshal([]byte(ev), &ch); err != nil {
			continue
		}
		if len(ch.Choices) > 0 && ch.Choices[0].FinishReason != nil && *ch.Choices[0].FinishReason != "" {
			finalChunk = ch
		}
	}
	if len(finalChunk.Choices) == 0 || finalChunk.Choices[0].FinishReason == nil || *finalChunk.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("final chunk finish_reason: want tool_calls, got %+v", finalChunk)
	}
}

func TestWriteAsSSE_PlainText(t *testing.T) {
	input := ollamaResp("Hello there!")
	// No tool calls, so transform is a no-op pass-through.
	transformed, _ := applyToolCallTransform(input)

	rr := httptest.NewRecorder()
	writeAsSSE(rr, transformed)

	events := parseSSEEvents(rr.Body.String())
	if len(events) == 0 || events[len(events)-1] != "[DONE]" {
		t.Fatalf("expected [DONE] as last event, got %v", events)
	}

	var contentSeen string
	for _, ev := range events {
		if ev == "[DONE]" {
			continue
		}
		var chunk sseChunk
		if err := json.Unmarshal([]byte(ev), &chunk); err != nil {
			t.Fatalf("invalid SSE JSON: %v", err)
		}
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != nil {
			contentSeen += *chunk.Choices[0].Delta.Content
		}
	}
	if contentSeen != "Hello there!" {
		t.Errorf("content: want %q, got %q", "Hello there!", contentSeen)
	}
}

func TestWriteAsSSE_MultipleToolCalls(t *testing.T) {
	input := ollamaResp(
		`<tool_call>{"name":"a","arguments":{}}</tool_call>` + "\n" +
			`<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>`,
	)
	transformed, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	writeAsSSE(rr, transformed)

	events := parseSSEEvents(rr.Body.String())
	names := map[string]bool{}
	for _, ev := range events {
		if ev == "[DONE]" {
			continue
		}
		var chunk sseChunk
		if err := json.Unmarshal([]byte(ev), &chunk); err != nil {
			t.Fatalf("invalid SSE JSON: %v", err)
		}
		if len(chunk.Choices) > 0 {
			for _, tc := range chunk.Choices[0].Delta.ToolCalls {
				if tc.Function != nil && tc.Function.Name != "" {
					names[tc.Function.Name] = true
				}
			}
		}
	}
	if !names["a"] || !names["b"] {
		t.Errorf("expected tool names a and b in SSE events, got %v", names)
	}
}

// --- proxyTransformInner (integration) ---

// fakeOllama returns an httptest.Server whose handler calls fn for each request.
func fakeOllama(fn func(callN int, w http.ResponseWriter)) *httptest.Server {
	callN := 0
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callN++
		w.Header().Set("Content-Type", "application/json")
		fn(callN, w)
	}))
}

func writeOllamaResp(w http.ResponseWriter, content string) {
	json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
		"id": "chatcmpl-test", "object": "chat.completion", "model": "test",
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": content},
			"finish_reason": "stop",
		}},
	})
}

// TestProxyTransform_EmptyRetryExtractsToolCall is the regression test for the case
// where the model returns empty on the first call (confused by tools), then on the
// retry generates <tool_call> XML in plain text. The retry must go through the
// transform path so the XML becomes proper SSE tool_calls deltas — not raw text.
func TestProxyTransform_EmptyRetryExtractsToolCall(t *testing.T) {
	upstream := fakeOllama(func(callN int, w http.ResponseWriter) {
		if callN == 1 {
			writeOllamaResp(w, "") // empty first response
		} else {
			// Retry: model generates <tool_call> in text even without tools in request.
			writeOllamaResp(w, `Bing blocked. Let me try another way.<tool_call>{"name":"web_search","arguments":{"query":"test"}}</tool_call>`)
		}
	})
	defer upstream.Close()

	originalBody := []byte(`{"model":"test","messages":[{"role":"user","content":"search"}],"tools":[{"type":"function","function":{"name":"web_search","parameters":{"type":"object"}}}],"stream":true}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(originalBody))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	proxyTransformInner(rr, req, upstream.URL, forceNoStream(originalBody), originalBody, false)

	body := rr.Body.String()

	// Must NOT contain raw <tool_call> XML — that would mean it was passed through as text.
	if strings.Contains(body, "<tool_call>") {
		t.Error("response must not contain raw <tool_call> XML; expected SSE tool_calls deltas")
	}

	// Must contain proper SSE tool_calls delta with the right tool name.
	events := parseSSEEvents(body)
	var toolCallName string
	for _, ev := range events {
		if ev == "[DONE]" {
			continue
		}
		var chunk sseChunk
		if err := json.Unmarshal([]byte(ev), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			for _, tc := range chunk.Choices[0].Delta.ToolCalls {
				if tc.Function != nil && tc.Function.Name != "" {
					toolCallName = tc.Function.Name
				}
			}
		}
	}
	if toolCallName != "web_search" {
		t.Errorf("expected SSE tool_calls delta with name=web_search, got %q", toolCallName)
	}
}

// TestTransform_PythonRegexEscapes verifies that \s, \d, \w inside Python regex
// strings in execute_code arguments are accepted (repaired to \\s etc.) rather
// than causing a JSON parse failure.
func TestTransform_PythonRegexEscapes(t *testing.T) {
	// Literal \s \d \w are invalid JSON escape sequences; the model emits them
	// verbatim inside string values when writing Python regex patterns.
	rawContent := "<tool_call>\n{\"name\": \"execute_code\", \"arguments\": {\"code\": \"import re\\nre.sub(r'\\s+', ' ', text)\\nre.findall(r'\\d+', s)\\nre.match(r'\\w+', t)\"}}\n</tool_call>"
	input := ollamaResp(rawContent)
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)

	if r.FinishReason != "tool_calls" {
		t.Errorf("finish_reason: want tool_calls, got %s — \\s/\\d/\\w not repaired", r.FinishReason)
	}
	if len(r.ToolCalls) != 1 {
		t.Fatalf("want 1 tool call, got %d", len(r.ToolCalls))
	}
	if r.ToolCalls[0].Function.Name != "execute_code" {
		t.Errorf("tool name: want execute_code, got %s", r.ToolCalls[0].Function.Name)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(r.ToolCalls[0].Function.Arguments), &args); err != nil {
		t.Fatalf("arguments not valid JSON after repair: %v", err)
	}
	if !strings.Contains(args["code"], `\s+`) {
		t.Errorf("expected \\s+ in code, got: %s", args["code"])
	}
}

// TestProxyTransform_EmptyRetryDoesNotLoop verifies that if both the original
// and the retry response are empty, we do not recurse infinitely — the second
// empty response is written through as-is.
func TestProxyTransform_EmptyRetryDoesNotLoop(t *testing.T) {
	calls := 0
	upstream := fakeOllama(func(callN int, w http.ResponseWriter) {
		calls++
		writeOllamaResp(w, "") // always empty
	})
	defer upstream.Close()

	originalBody := []byte(`{"model":"test","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"stream":true}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(originalBody))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	proxyTransformInner(rr, req, upstream.URL, forceNoStream(originalBody), originalBody, false)

	if calls != 2 {
		t.Errorf("expected exactly 2 upstream calls (original + one retry), got %d", calls)
	}
}

// TestProxyTransform_NoRetryAfterToolResults verifies that an empty response from
// the model is NOT retried when the conversation history already contains role:"tool"
// messages. Stripping tools from such a history creates a mangled context that makes
// the model hallucinate partial <tool_call> fragments.
func TestProxyTransform_NoRetryAfterToolResults(t *testing.T) {
	calls := 0
	upstream := fakeOllama(func(callN int, w http.ResponseWriter) {
		calls++
		writeOllamaResp(w, "") // empty response after tool result
	})
	defer upstream.Close()

	// originalBody has a role:"tool" message in history — tools have already been dispatched.
	originalBody := []byte(`{"model":"test","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"result"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"stream":true}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(originalBody))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	proxyTransformInner(rr, req, upstream.URL, forceNoStream(originalBody), originalBody, false)

	if calls != 1 {
		t.Errorf("expected exactly 1 upstream call (no retry after tool results), got %d", calls)
	}
}
