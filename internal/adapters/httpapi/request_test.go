package httpapi

import "testing"

// Ported from TestRequestedStream_{True,False,Absent}: the stream flag now lives
// on the parsed request.
func TestParseRequest_Stream(t *testing.T) {
	cases := map[string]bool{
		`{"model":"x","stream":true}`:  true,
		`{"model":"x","stream":false}`: false,
		`{"model":"x"}`:                false,
	}
	for body, want := range cases {
		req, err := parseRequest([]byte(body))
		if err != nil {
			t.Fatalf("%s: %v", body, err)
		}
		if req.Stream != want {
			t.Errorf("%s: stream = %v, want %v", body, req.Stream, want)
		}
	}
}

func TestParseRequest_Fields(t *testing.T) {
	body := `{"model":"m","stream":true,"stream_options":{"include_usage":true},
		"max_completion_tokens":50,"temperature":0.5,"stop":"END",
		"messages":[
			{"role":"system","content":"be brief"},
			{"role":"user","content":[{"type":"text","text":"hello "},{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"world"}]},
			{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"c1","content":"42"}
		]}`
	req, err := parseRequest([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if !req.IncludeUsage || req.MaxTokens == nil || *req.MaxTokens != 50 || *req.Temperature != 0.5 {
		t.Errorf("scalar fields wrong: %+v", req)
	}
	if len(req.Stop) != 1 || req.Stop[0] != "END" {
		t.Errorf("stop: %v", req.Stop)
	}
	if len(req.Messages) != 4 {
		t.Fatalf("messages: %d", len(req.Messages))
	}
	if got := req.Messages[1].Content; got != "hello world" {
		t.Errorf("text parts should concatenate, skipping images: %q", got)
	}
	if tc := req.Messages[2].ToolCalls; len(tc) != 1 || tc[0].Name != "f" {
		t.Errorf("tool calls: %+v", tc)
	}
	if req.Messages[3].ToolCallID != "c1" || req.Messages[3].Content != "42" {
		t.Errorf("tool message: %+v", req.Messages[3])
	}
	if string(req.Raw) != body {
		t.Error("Raw must be the original body, byte for byte")
	}
}

// Only a non-object body, or n > 1, is rejected.
func TestParseRequest_Invalid(t *testing.T) {
	for _, body := range []string{`not json`, `{"model":"m","n":2}`} {
		if _, err := parseRequest([]byte(body)); err == nil {
			t.Errorf("%s: want error", body)
		}
	}
}

// Critique pass 1, finding 7: bodies the original proxy forwarded untouched
// must still be accepted.
func TestParseRequest_Lenient(t *testing.T) {
	cases := []string{
		`{"model":"m","max_tokens":1024.0}`,
		`{"model":"m","n":1}`,
		`{"model":"m","stop":7}`,
		`{"model":"m","messages":[{"role":"user","content":42}]}`,
		`{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"c","function":{"name":"f","arguments":{"q":"x"}}}]}]}`,
	}
	for _, body := range cases {
		req, err := parseRequest([]byte(body))
		if err != nil {
			t.Errorf("%s: want accepted, got %v", body, err)
			continue
		}
		if string(req.Raw) != body {
			t.Errorf("%s: Raw must be forwarded unchanged", body)
		}
	}
	req, _ := parseRequest([]byte(cases[0]))
	if req.MaxTokens == nil || *req.MaxTokens != 1024 {
		t.Errorf("1024.0 should read as 1024, got %v", req.MaxTokens)
	}
	req, _ = parseRequest([]byte(cases[4]))
	if got := req.Messages[0].ToolCalls[0].Arguments; got != `{"q":"x"}` {
		t.Errorf("object arguments should become their JSON text, got %q", got)
	}
}
