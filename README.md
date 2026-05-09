# llmux

A small HTTP proxy in front of vLLM and Ollama that repairs malformed tool calls and strips `<think>` blocks before returning responses to clients.

## Problem

Tool-calling models in vLLM and Ollama frequently return malformed JSON tool calls — `