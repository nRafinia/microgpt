# microgpt.cs

A C# port of [Andrej Karpathy's microgpt](https://karpathy.github.io/2026/02/12/microgpt/) — "the most atomic way to train and run inference for a GPT" — implemented twice:

1. **`MicroGptScalar`** — a faithful, line-by-line port of the original scalar autograd design (the `Value` class computation graph).
2. **`MicroGptVector`** — a vectorized rewrite with hand-derived analytic gradients, verified **bit-for-bit identical** to the scalar version.

Both are single-file, dependency-free console apps. No ML libraries, no NuGet packages — just the .NET BCL.

## What it does

Trains a GPT-2-style transformer (1 layer, 4 attention heads, 16-dim embeddings, **4,192 parameters**) on ~32,000 names, then samples new, plausible-sounding names:

```
num docs: 32033
vocab size: 27
num params: 4192
step    1 / 1000 | loss 3.4077
...
step 1000 / 1000 | loss 2.1829

--- inference (new, hallucinated names) ---
sample  1: aaman
sample  2: welara
sample  3: kaman
sample  4: alore
...
```

The full pipeline is included: dataset download, character-level tokenizer with a BOS token, autograd, multi-head attention with a live KV cache (backpropagated through, as in the original), RMSNorm, residual connections, cross-entropy loss, Adam with linear learning-rate decay, and temperature-based sampling.

## Benchmarks

1,000 training steps + 20 inference samples, same machine, Release build:

| Implementation | Runtime | Speedup vs Python |
|---|---:|---:|
| Python (original `microgpt.py`, per Karpathy's blog) | ~60 s | 1× |
| `MicroGptScalar` on .NET 8 | 19.7 s | ~3× |
| `MicroGptScalar` on .NET 10 | 18.4 s | ~3.3× |
| `MicroGptVector` on .NET 10 | **0.65 s** | **~90×** |

Two takeaways:

- **The language switch alone gives ~3×** (JIT vs interpreter), but no more — the scalar version's bottleneck is not arithmetic, it's the autograd graph itself: millions of `Value` node allocations per training step, GC pressure, pointer chasing, and a topological sort over the whole graph on every backward pass. That's also why .NET 10 only improves it by ~7%: the JIT's new escape analysis can't stack-allocate nodes that live on in the graph.
- **Changing the autograd granularity gives ~28×** on top of that. The vectorized version replaces the graph of scalars with plain `double[]` activations and an analytic backward pass per layer, reducing the graph from hundreds of thousands of nodes per step to zero.

## Correctness verification

The vectorized version is not "approximately" correct — running both projects with the same seed produces **bit-for-bit identical output**: all 1,000 loss values match to full double precision, and all 20 generated names are the same.

```
$ diff scalar_output.txt vector_output.txt
(no differences)
```

This works because:

- Both versions consume the RNG in the same order (Fisher–Yates shuffle, Box–Muller Gaussian init, cumulative-sum sampling).
- Floating-point summation order is deliberately preserved everywhere (FP addition is not associative).
- Every hand-derived gradient is exact. The formulas:

| Layer | Forward | Backward |
|---|---|---|
| Linear | `y = Wx` | `dW += dy ⊗ x`, `dx = Wᵀ dy` |
| RMSNorm | `y = x·s`, `s = (mean(x²)+ε)^-½` | `dxᵢ = s·dyᵢ − xᵢ·s³·(Σⱼ dyⱼxⱼ)/n` |
| Softmax + cross-entropy | `L = −log p[target]` | `dlogits = p − onehot` |
| Softmax (attention) | `a = softmax(z)` | `dzₜ = aₜ·(daₜ − Σ a·da)` |
| ReLU | `max(0, x)` | `dx = dy · 1[x > 0]` |
| Attention (causal, KV cache) | per-head QKᵀ/√d → softmax → ·V | reverse-order BPTT; `dK[t]`/`dV[t]` accumulate contributions from all later positions before being consumed at position `t` |

The bit-exact match is the strongest possible check: a single wrong term in any backward formula would diverge the entire training trajectory within a few Adam steps.

> **Note:** if you push vectorization further with SIMD (`TensorPrimitives`, `Vector<double>`), tree-shaped reductions change the FP summation order. Results then differ in the last bits and amplify over training — statistically equivalent (same loss curve, same quality), but no longer bit-identical. That's an inherent property of floating point, not a logic error; PyTorch behaves the same way across GPUs.

## Project structure

```
├── MicroGptScalar/     # faithful port: Value class, operator overloading, autograd graph
│   └── Program.cs
├── MicroGptVector/     # vectorized: double[] tensors, manual analytic backprop
│   └── Program.cs
└── README.md
```

## Running

Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download) (also builds on .NET 8 by changing `TargetFramework`, but .NET 8 support ends November 2026).

```bash
cd MicroGptVector        # or MicroGptScalar
dotnet run -c Release
```

On first run, the names dataset (`input.txt`) is downloaded automatically from the [makemore](https://github.com/karpathy/makemore) repository.

## Notes on the scalar port

- The `Value` class mirrors the Python original: `Data`, `Grad`, children, and local gradients per node, with C# operator overloading (`+`, `-`, `*`, `/`, `Pow`, `Log`, `Exp`, `Relu`) and an implicit `double → Value` conversion standing in for Python's `__radd__`/`__rmul__` reflected operators.
- The only deliberate deviation: `Backward()` builds the topological order with an explicit stack (iterative DFS) instead of recursion, avoiding `StackOverflowException` on deep graphs. The resulting order — and therefore the numerics — is identical.
- Node identity uses `ReferenceEqualityComparer`, matching Python's object-identity semantics in the `visited` set.
- `random.gauss`, `random.shuffle`, and `random.choices` are reimplemented via Box–Muller, Fisher–Yates, and cumulative-sum sampling respectively. Since .NET's `Random` differs from CPython's Mersenne Twister, results don't match the Python original bit-for-bit — but the two C# versions match each other exactly.

## Credits

The algorithm, architecture, and original implementation are by [Andrej Karpathy](https://github.com/karpathy):

- Blog post: [microgpt](https://karpathy.github.io/2026/02/12/microgpt/)
- Original source: [microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

This repository is an educational port; the scalar version preserves the original's pedagogical clarity, and the vectorized version shows what "everything else is just efficiency" looks like in practice.
