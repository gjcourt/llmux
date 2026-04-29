package main

import (
	"encoding/json"
	"testing"
)

// --- hasTools ---

func TestHasTools_WithTools(t *testing.T) {
	body := `{"model":"x","tools":[{"type":"function","function":{"name":"f"}}]}`
	if !hasTools([]byte(body)) {
		t.Fatal("expected hasTools=true")
	}
}

func TestHasTools_EmptyTools(t *testing.T) {
	body := `{"model":"x","tools":[]}`
	if hasTools([]byte(body)) {
		t.Fatal("expected hasTools=false for empty tools array")
	}
}

func TestHasTools_NoTools(t *testing.T) {
	body := `{"model":"x","messages":[]}`
	if hasTools([]byte(body)) {
		t.Fatal("expected hasTools=false when tools key absent")
	}
}

// --- forceNoStream ---

func TestForceNoStream_AddsField(t *testing.T) {
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

func TestTransform_MultipleToolCalls(t *testing.T) {
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
	// Model says something before calling a tool.
	input := ollamaResp("Let me look that up.\n<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>")
	out, err := applyToolCallTransform(input)
	if err != nil {
		t.Fatal(err)
	}
	r := parseTransform(t, out)

	if r.Content == nil || *r.Content != "Let me look that up." {
		t.Errorf("want preamble content, got %v", r.Content)
	}
	if len(r.ToolCalls) != 1 {
		t.Errorf("want 1 tool call, got %d", len(r.ToolCalls))
	}
}

func TestTransform_NoToolCall_PassThrough(t *testing.T) {
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
	_, err := applyToolCallTransform([]byte("not json"))
	if err == nil {
		t.Fatal("expected error for invalid JSON input")
	}
}

// --- stripTools ---

func TestStripTools_RemovesTools(t *testing.T) {
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

func TestIsEmptyNonToolResponse_EmptyContent(t *testing.T) {
	if !isEmptyNonToolResponse(ollamaResp("")) {
		t.Fatal("empty content should be detected as empty")
	}
}

func TestIsEmptyNonToolResponse_PlainText(t *testing.T) {
	if isEmptyNonToolResponse(ollamaResp("Hello!")) {
		t.Fatal("non-empty content should not be empty")
	}
}

func TestIsEmptyNonToolResponse_WithToolCalls(t *testing.T) {
	// A properly transformed response with tool_calls should never be "empty".
	input := ollamaResp(`<tool_call>{"name":"f","arguments":{}}</tool_call>`)
	transformed, _ := applyToolCallTransform(input)
	if isEmptyNonToolResponse(transformed) {
		t.Fatal("response with tool_calls should not be considered empty")
	}
}
