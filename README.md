# Rinha de Backend 2026 — `gustavoqnt-rust`

Backend de detecção de fraude para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026), focado em latência mínima e zero alocação no caminho quente.

## Arquitetura

```
client → nginx (UDS load balancer) ─┬→ app1 (Rust + monoio)
                                    └→ app2 (Rust + monoio)
```

- **2 instâncias** do servidor Rust atrás de **nginx**, comunicação via **Unix Domain Sockets** (sem TCP loopback entre LB e app).
- **`monoio`** como runtime async com driver **io_uring** (kernel ≥ 5.11). Cada instância roda um único worker single-threaded → zero contention dentro do processo.
- Servidor HTTP/1.1 escrito à mão (sem framework): parser de request manual, respostas pré-construídas, write em um único syscall por request.

## kNN

- Índice **IVF (Inverted File)** com **K=4096 clusters**, vetores **i16 quantizados** (scale=10000), `dim=14`.
- Layout **block-SoA de 8 slots**: cada bloco guarda 8 refs transpostos em SoA-por-bloco, permitindo distância L2 vetorizada com **AVX2** sobre 8 candidatos em paralelo.
- Padding por `i16::MAX` nos slots vagos do último bloco — distância gigante, nunca entra no top-5.
- **Top-5 com insertion-sort** e tie-break determinístico por índice original.

### Two-pass adaptive nprobe

Cada query roda primeiro com `fast_nprobe=8` (escaneia ~5.8k vetores de 3M). Se o bucket do top-5 cair em zona de incerteza (qualquer valor ≠ 0 e ≠ 5), escala automaticamente pra `full_nprobe=32` (~23k vetores) e responde com esse resultado. Resultado:

- 100% dos casos clear-legit (bucket=0) e clear-fraud (bucket=5) terminam no fast path — latência mínima.
- Casos boundary recebem precisão de scan maior — zero falsos negativos.

## Otimizações de hot path

- **Zero allocation por request**: buffer de leitura reutilizado, top-5 em stack, distâncias em registradores AVX2.
- **Respostas pré-construídas**: 6 buffers HTTP/1.1 (200 OK com `approved` + `fraud_score`) materializados no boot. Resposta = `write_all(&'static [u8])`.
- **Parser manual** de JSON: extrai os 12 campos necessários sem alocar string intermediária, escaneia bytes diretamente do buffer da request.
- **`mimalloc`** como global allocator.
- Build com **`target-cpu=haswell`** (Mac Mini Late 2014 da banca usa Intel Haswell) → AVX2/FMA/BMI2 estáticos.
- `lto=fat`, `codegen-units=1`, symbols strip.

## Recursos

| Service | CPU  | Memory |
| ------- | ---: | -----: |
| app1    | 0.40 | 150 MB |
| app2    | 0.40 | 150 MB |
| nginx   | 0.20 |  50 MB |
| **Total** | **1.00** | **350 MB** |

## Endpoints

- `POST /fraud-score` — recebe a transação, devolve `{ "approved": bool, "fraud_score": number }`.
- `GET /ready` — health-check.

## Stack

`rust` · `monoio` · `io_uring` · `avx2` · `nginx` · `unix domain sockets` · `mimalloc`

## Repo

Código fonte: [`main` branch](../../tree/main).
