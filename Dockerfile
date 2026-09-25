FROM --platform=$BUILDPLATFORM golang:1.26-alpine AS builder
ARG TARGETOS
ARG TARGETARCH
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -trimpath -ldflags="-s -w" -o /llmux ./cmd/llmux

FROM alpine:3.24
# ca-certificates: the Anthropic provider calls https://api.anthropic.com.
RUN apk add --no-cache ca-certificates tzdata
COPY --from=builder /llmux /usr/local/bin/llmux
EXPOSE 8080
USER 65534:65534
ENTRYPOINT ["llmux"]
