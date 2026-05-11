FROM rust:1.86-slim-bookworm AS builder

WORKDIR /build

COPY native/rinha-server/Cargo.toml native/rinha-server/Cargo.lock ./native/rinha-server/
COPY native/rinha-server/src ./native/rinha-server/src

RUN cargo build --release --manifest-path native/rinha-server/Cargo.toml --bin rinha-server

FROM debian:bookworm-slim

WORKDIR /app

ENV IVF_PATH=/app/resources/references.ivf.bin
ENV IVF_BLOCKS_PATH=/app/resources/references.ivf-blocks-K4096.bin
ENV IVF_NPROBE=32
ENV SOCK_PATH=/run/rinha/rinha.sock

COPY --from=builder /build/native/rinha-server/target/release/rinha-server /app/rinha-server
COPY resources/references.ivf.bin /app/resources/references.ivf.bin
COPY resources/references.ivf-blocks.bin /app/resources/references.ivf-blocks.bin
COPY resources/references.ivf-blocks-K4096.bin /app/resources/references.ivf-blocks-K4096.bin

CMD ["/app/rinha-server"]
