package anthropic

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/gjcourt/llmux/internal/domain"
)

type recorder struct{ events []domain.Event }

func (r *recorder) Emit(e domain.Event) error {
	r.events = append(r.events, e)
	return nil
}

func (r *recorder) text() string {
	var b strings.Builder
	for _, e := range r.events {
		if e.Kind == domain.EventText {
			b.WriteString(e.Text)
		}
	}
	return b.String()
}

func (r *recorder) kinds(k domain.EventKind) []domain.Event {
	var out []domain.Event
	for _, e := range r.events {
		if e.Kind == k {
			out = append(out, e)
		}
	}
	return out
}

func parseFixture(t *testing.T, name string) *recorder {
	t.Helper()
	f, err := os.Open("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rec := &recorder{}
	if err := parseStream(f, rec, 1700000000); err != nil {
		t.Fatalf("parseStream: %v", err)
	}
	return rec
}

// testdata/plain.sse is a real claude-sonnet-5 stream captured 2026-09-25.
func TestParseStream_Plain(t *testing.T) {
	rec := parseFixture(t, "plain.sse")
	if len(rec.events) == 0 || rec.events[0].Kind != domain.EventStart {
		t.Fatalf("first event must be Start: %+v", rec.events)
	}
	start := rec.events[0]
	if start.ID != "msg_011CfPd6DCknge5HVTivUBdY" || start.Model != "claude-sonnet-5" || start.Created != 1700000000 {
		t.Errorf("start: %+v", start)
	}
	if got := rec.text(); got != "Red and blue." {
		t.Errorf("text = %q", got)
	}
	fin := rec.kinds(domain.EventFinish)
	if len(fin) != 1 || fin[0].FinishReason != "stop" {
		t.Errorf("finish: %+v", fin)
	}
	us := rec.kinds(domain.EventUsage)
	if len(us) != 1 || us[0].Usage != (domain.Usage{PromptTokens: 20, CompletionTokens: 8, TotalTokens: 28}) {
		t.Errorf("usage: %+v", us)
	}
	if last := rec.events[len(rec.events)-1]; last.Kind != domain.EventUsage {
		t.Errorf("usage must come last, got %v", last.Kind)
	}
}

// testdata/websearch.sse is a real stream with thinking, server_tool_use and
// web_search_tool_result blocks between the text blocks. Only the text is
// relayed; websearch.text is that text, extracted independently.
func TestParseStream_DropsNonTextBlocks(t *testing.T) {
	rec := parseFixture(t, "websearch.sse")
	want, err := os.ReadFile("testdata/websearch.text")
	if err != nil {
		t.Fatal(err)
	}
	if got := rec.text(); got != string(want) {
		t.Errorf("text mismatch:\n got %q\nwant %q", got, want)
	}
	if tc := rec.kinds(domain.EventToolCall); len(tc) != 0 {
		t.Errorf("server tool use must never surface as a tool call: %+v", tc)
	}
	if strings.Contains(rec.text(), "encrypted") {
		t.Error("search result content leaked into text")
	}
}

// message_delta's usage is cumulative; its input_tokens includes the search
// results and must win over message_start's.
func TestParseStream_UsageFromMessageDelta(t *testing.T) {
	rec := parseFixture(t, "websearch.sse")
	us := rec.kinds(domain.EventUsage)
	if len(us) != 1 || us[0].Usage != (domain.Usage{PromptTokens: 29268, CompletionTokens: 396, TotalTokens: 29664}) {
		t.Errorf("usage: %+v", us)
	}
}

func sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("data: " + e + "\n\n")
	}
	return b.String()
}

const (
	evStart = `{"type":"message_start","message":{"id":"m1","model":"claude-x","usage":{"input_tokens":5,"cache_creation_input_tokens":2,"cache_read_input_tokens":3,"output_tokens":1}}}`
	evText  = `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	evDelta = `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`
	evStop  = `{"type":"message_stop"}`
)

func TestParseStream_CacheTokensCountAsPrompt(t *testing.T) {
	body := sse(evStart, evText, evDelta,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`, evStop)
	rec := &recorder{}
	if err := parseStream(strings.NewReader(body), rec, 0); err != nil {
		t.Fatal(err)
	}
	us := rec.kinds(domain.EventUsage)
	if len(us) != 1 || us[0].Usage != (domain.Usage{PromptTokens: 10, CompletionTokens: 7, TotalTokens: 17}) {
		t.Errorf("usage without input in message_delta should keep start's: %+v", us)
	}
}

func TestParseStream_FinishReasons(t *testing.T) {
	cases := map[string]string{
		"end_turn": "stop", "stop_sequence": "stop", "pause_turn": "length",
		"max_tokens": "length", "refusal": "content_filter", "tool_use": "tool_calls",
	}
	for stop, want := range cases {
		body := sse(evStart, evText, evDelta, `{"type":"message_delta","delta":{"stop_reason":"`+stop+`"},"usage":{"output_tokens":1}}`, evStop)
		rec := &recorder{}
		if err := parseStream(strings.NewReader(body), rec, 0); err != nil {
			t.Fatal(err)
		}
		if fin := rec.kinds(domain.EventFinish); len(fin) != 1 || fin[0].FinishReason != want {
			t.Errorf("%s: got %+v, want %s", stop, fin, want)
		}
	}
}

// A stream cut off before message_stop is an error, never a complete answer.
func TestParseStream_TruncatedIsError(t *testing.T) {
	rec := &recorder{}
	err := parseStream(strings.NewReader(sse(evStart, evText, evDelta)), rec, 0)
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("want ErrUnexpectedEOF, got %v", err)
	}
	if len(rec.kinds(domain.EventFinish)) != 0 {
		t.Error("a truncated stream must not emit Finish")
	}
}

func TestParseStream_ErrorEvent(t *testing.T) {
	body := sse(evStart, evText, evDelta, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	err := parseStream(strings.NewReader(body), &recorder{}, 0)
	var se *StreamError
	if !errors.As(err, &se) || se.Type != "overloaded_error" || se.Message != "Overloaded" {
		t.Fatalf("got %v", err)
	}
}

// A delta for a non-text block index must not leak even if it claims to be
// text_delta.
func TestParseStream_IgnoresDeltasForNonTextBlocks(t *testing.T) {
	body := sse(evStart,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"secret"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"leak"}}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}`,
		evStop)
	rec := &recorder{}
	if err := parseStream(strings.NewReader(body), rec, 0); err != nil {
		t.Fatal(err)
	}
	if got := rec.text(); got != "answer" {
		t.Errorf("text = %q", got)
	}
}

type failingSink struct{ after int }

func (f *failingSink) Emit(domain.Event) error {
	if f.after == 0 {
		return errors.New("client gone")
	}
	f.after--
	return nil
}

// When the client goes away the parser stops and reports it.
func TestParseStream_SinkErrorStops(t *testing.T) {
	err := parseStream(strings.NewReader(sse(evStart, evText, evDelta, evDelta, evStop)), &failingSink{after: 1}, 0)
	if err == nil || err.Error() != "client gone" {
		t.Fatalf("got %v", err)
	}
}

func TestParseStream_MalformedEvent(t *testing.T) {
	if err := parseStream(strings.NewReader("data: {not json\n\n"), &recorder{}, 0); err == nil {
		t.Fatal("want error")
	}
}

// The captured web-search stream cites one source twice; it is reported once.
func TestParseStream_Citations(t *testing.T) {
	rec := parseFixture(t, "websearch.sse")
	cites := rec.kinds(domain.EventCitation)
	if len(cites) == 0 {
		t.Fatal("want citations from the web search fixture")
	}
	seen := map[string]bool{}
	for _, c := range cites {
		if c.Citation.URL == "" || !strings.HasPrefix(c.Citation.URL, "https://") {
			t.Errorf("citation without a URL: %+v", c.Citation)
		}
		if seen[c.Citation.URL] {
			t.Errorf("citation repeated: %s", c.Citation.URL)
		}
		seen[c.Citation.URL] = true
	}
	if len(cites) != 1 {
		t.Errorf("want 1 deduped citation, got %d", len(cites))
	}
	if rec.events[0].Kind != domain.EventStart {
		t.Error("Start must come first")
	}
}

const evCite = `{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://a.example/x","title":"A","cited_text":"...","encrypted_index":"E1"}}}`

func TestParseStream_CitationDedupedByURL(t *testing.T) {
	body := sse(evStart, evText, evCite, evDelta, evCite,
		`{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://b.example/","title":""}}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`, evStop)
	rec := &recorder{}
	if err := parseStream(strings.NewReader(body), rec, 0); err != nil {
		t.Fatal(err)
	}
	cites := rec.kinds(domain.EventCitation)
	if len(cites) != 2 || cites[0].Citation != (domain.Citation{URL: "https://a.example/x", Title: "A"}) || cites[1].Citation.URL != "https://b.example/" {
		t.Errorf("citations: %+v", cites)
	}
}

// A paused turn's blocks must round-trip exactly what the API needs to
// resume: text with its citations, thinking with its signature, and the
// server tool's input assembled from its JSON deltas.
func TestParseTurn_AccumulatesBlocks(t *testing.T) {
	body := sse(evStart,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"search"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\": "}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"go 1.26\"}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://go.dev","title":"Go","encrypted_content":"ENC"}]}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"content_block_start","index":3,"content_block":{"citations":[],"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://go.dev","title":"Go","encrypted_index":"EI"}}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"Go 1.26 is out."}}`,
		`{"type":"content_block_stop","index":3}`,
		`{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":9}}`, evStop)
	st := newStreamState(0)
	rec := &recorder{}
	tr, err := st.parseTurn(strings.NewReader(body), rec)
	if err != nil {
		t.Fatal(err)
	}
	if tr.stopReason != "pause_turn" || len(tr.blocks) != 4 {
		t.Fatalf("turn: %s, %d blocks", tr.stopReason, len(tr.blocks))
	}
	want := []string{
		`{"signature":"SIG","thinking":"let me search","type":"thinking"}`,
		`{"id":"srvtoolu_1","input":{"query":"go 1.26"},"name":"web_search","type":"server_tool_use"}`,
		`{"content":[{"encrypted_content":"ENC","title":"Go","type":"web_search_result","url":"https://go.dev"}],"tool_use_id":"srvtoolu_1","type":"web_search_tool_result"}`,
		`{"citations":[{"type":"web_search_result_location","url":"https://go.dev","title":"Go","encrypted_index":"EI"}],"text":"Go 1.26 is out.","type":"text"}`,
	}
	for i, w := range want {
		if string(tr.blocks[i]) != w {
			t.Errorf("block %d:\n got %s\nwant %s", i, tr.blocks[i], w)
		}
	}
	if rec.text() != "Go 1.26 is out." {
		t.Errorf("text: %q", rec.text())
	}
	if len(rec.kinds(domain.EventFinish)) != 0 {
		t.Error("parseTurn must leave Finish to the caller")
	}
}

func TestParseTurn_InvalidToolInput(t *testing.T) {
	body := sse(evStart,
		`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"s","name":"web_search","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\": "}}`,
		`{"type":"content_block_stop","index":0}`, evStop)
	if _, err := newStreamState(0).parseTurn(strings.NewReader(body), &recorder{}); err == nil {
		t.Fatal("want error for truncated tool input")
	}
}

// An error event before message_start has reached nobody, so it becomes an
// upstream error with a status that says whether to retry.
func TestParseStream_ErrorBeforeStartIsStatus(t *testing.T) {
	for typ, want := range map[string]int{"overloaded_error": 529, "rate_limit_error": 429, "api_error": 500, "other": 502} {
		body := sse(`{"type":"error","error":{"type":"` + typ + `","message":"m"}}`)
		err := parseStream(strings.NewReader(body), &recorder{}, 0)
		var ue *domain.UpstreamError
		if !errors.As(err, &ue) || ue.Status != want || !strings.Contains(string(ue.Body), typ) {
			t.Errorf("%s: got %v", typ, err)
		}
	}
}

// Current API: message_delta carries the full cumulative usage, cache fields
// included.
func TestParseStream_CacheTokensFromMessageDelta(t *testing.T) {
	body := sse(evStart, evText, evDelta,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":50,"cache_creation_input_tokens":20,"cache_read_input_tokens":30,"output_tokens":7}}`, evStop)
	rec := &recorder{}
	if err := parseStream(strings.NewReader(body), rec, 0); err != nil {
		t.Fatal(err)
	}
	if us := rec.kinds(domain.EventUsage); len(us) != 1 || us[0].Usage != (domain.Usage{PromptTokens: 100, CompletionTokens: 7, TotalTokens: 107}) {
		t.Errorf("usage: %+v", us)
	}
}

// On the real capture, every replayed block equals what the API sent, with
// the deltas folded in: results and tool-use starts unchanged apart from the
// rebuilt input, thinking keeping its signature, text keeping citations.
func TestParseTurn_RealCaptureRoundTrip(t *testing.T) {
	f, err := os.Open("testdata/websearch.sse")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tr, err := newStreamState(0).parseTurn(f, &recorder{})
	if err != nil {
		t.Fatal(err)
	}
	if tr.lossy {
		t.Error("the real capture must not be lossy")
	}
	types := []string{}
	for _, raw := range tr.blocks {
		var b map[string]any
		if err := json.Unmarshal(raw, &b); err != nil {
			t.Fatal(err)
		}
		typ, _ := b["type"].(string)
		types = append(types, typ)
		switch typ {
		case "thinking":
			if sig, _ := b["signature"].(string); len(sig) < 100 {
				t.Errorf("thinking lost its signature: %q", sig)
			}
		case "server_tool_use":
			in, _ := b["input"].(map[string]any)
			if q, _ := in["query"].(string); q == "" {
				t.Errorf("tool input not rebuilt: %v", b["input"])
			}
		case "web_search_tool_result":
			if c, _ := b["content"].([]any); len(c) == 0 {
				t.Error("search results lost")
			}
		}
	}
	if len(types) != 11 || types[0] != "thinking" || types[1] != "server_tool_use" || types[2] != "web_search_tool_result" {
		t.Errorf("block order: %v", types)
	}
	var cited int
	for _, raw := range tr.blocks {
		if strings.Contains(string(raw), "encrypted_index") {
			cited++
		}
	}
	if cited == 0 {
		t.Error("text citations (with encrypted_index) must be kept for replay")
	}
}
