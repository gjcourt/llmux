package openaicompat

// The stages in this file are pure: bytes in, bytes out, no I/O and no globals
// beyond compiled regexps. They were moved verbatim from the original main.go
// and keep its invariants — a clean tool-call response round-trips unchanged,
// and nil/empty input never panics.

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
)

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

// hasToolResultMessages returns true when the request body's messages array
// contains any role:"tool" message, indicating a tool has already been
// dispatched and its results injected. Retrying as plain chat in that state
// produces a mangled history (tool_calls without definitions) that confuses
// the model.
// hasToolResultMessages returns true when the request body's messages array
// contains any role:"tool" message, indicating a tool has already been
// dispatched and its results injected. Retrying as plain chat in that state
// produces a mangled history (tool_calls without definitions) that confuses
// the model.
func hasToolResultMessages(body []byte) bool {
	var req struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	for _, m := range req.Messages {
		if m.Role == "tool" {
			return true
		}
	}
	return false
}

// isEmptyNonToolResponse returns true when the model responded with no content and no tool calls —
// meaning the model silently gave up rather than answering or calling a tool.
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
			// Use Decoder instead of Unmarshal so trailing garbage after a
			// valid JSON object (comments, Python continuations, etc.) is ignored.
			tryDecode := func(s string) error {
				return json.NewDecoder(strings.NewReader(s)).Decode(&tc)
			}
			if err := tryDecode(raw); err != nil {
				// Pass 1: escape control characters and invalid escape sequences.
				repaired := repairJSONStrings(raw)
				if err2 := tryDecode(repaired); err2 != nil {
					// Pass 2: use json.SyntaxError offset to find and escape unescaped "
					// characters that terminate JSON strings early. Models embed Python
					// string literals like "\"text\"" without escaping their outer delimiters.
					repaired2 := repairUnescapedQuotes(repaired)
					if err3 := tryDecode(repaired2); err3 != nil {
						slog.Warn("failed to parse tool_call JSON", "err", err3, "raw", raw[:min(len(raw), 600)])
						continue
					}
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

// repairJSONStrings fixes two classes of model-generated JSON errors inside string values:
//  1. Literal control characters (newlines, tabs, carriage returns) → escape them.
//  2. Invalid escape sequences (e.g. \s \d \w from Python regex) → double the backslash
//     so \s becomes \\s, which JSON decodes back to \s (the regex metachar).
func repairJSONStrings(s string) string {
	var b strings.Builder
	inString := false
	escaped := false
	b.Grow(len(s))
	for _, c := range s {
		if escaped {
			escaped = false
			if inString {
				switch c {
				case '"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u':
					b.WriteRune(c)
				default:
					// Invalid escape (\s \d \w …) — double the backslash.
					b.WriteRune('\\')
					b.WriteRune(c)
				}
			} else {
				b.WriteRune(c)
			}
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

// repairUnescapedQuotes handles a third class of model error: Python string
// literals like "\"text\"" embedded in a JSON string value whose outer "
// delimiters are not escaped. The json.SyntaxError offset pinpoints exactly
// where the parse failed; we walk backward to find the unescaped " and escape
// it, then retry. Up to 20 passes handle multiple such quotes per string.
func repairUnescapedQuotes(s string) string {
	for range 20 {
		var se *json.SyntaxError
		err := json.NewDecoder(strings.NewReader(s)).Decode(new(any))
		if err == nil {
			return s
		}
		if !errors.As(err, &se) {
			return s
		}
		pos := int(se.Offset) - 1 // 0-indexed position of the offending char
		if pos < 0 || pos >= len(s) {
			return s
		}
		// Walk backward from the error position to find the unescaped " whose
		// premature string-close caused the parse failure.
		quotePos := -1
		for i := pos - 1; i >= 0; i-- {
			if s[i] != '"' {
				continue
			}
			// Count consecutive preceding backslashes.
			numBS := 0
			for j := i - 1; j >= 0 && s[j] == '\\'; j-- {
				numBS++
			}
			if numBS%2 == 0 {
				// Unescaped quote — this is our culprit.
				quotePos = i
				break
			}
			// Escaped quote — keep looking.
		}
		if quotePos < 0 {
			return s
		}
		s = s[:quotePos] + `\"` + s[quotePos+1:]
	}
	return s
}
