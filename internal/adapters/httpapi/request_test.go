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

func TestParseRequest_Invalid(t *testing.T) {
	for _, body := range []string{`not json`, `{"messages":[{"role":"user","content":42}]}`, `{"stop":7}`} {
		if _, err := parseRequest([]byte(body)); err == nil {
			t.Errorf("%s: want error", body)
		}
	}
}
