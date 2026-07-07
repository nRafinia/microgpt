namespace MicroGptVector;

public static class Program
{
    private const int NLayer = 1;
    private const int NEmbd = 16;
    private const int BlockSize = 16;
    private const int NHead = 4;
    private const int HeadDim = NEmbd / NHead;

    private static readonly Random Rng = new Random(42);

    private static double Gauss(double mu, double sigma)
    {
        var u1 = 1.0 - Rng.NextDouble();
        var u2 = Rng.NextDouble();
        return mu + sigma * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private static Tensor2 Matrix(int nout, int nin, double std = 0.08)
    {
        var t = new Tensor2(nout, nin);
        for (var i = 0; i < t.Data.Length; i++) t.Data[i] = Gauss(0, std);
        return t;
    }

    // y = W x   (forward)
    private static double[] Linear(double[] x, Tensor2 w)
    {
        var y = new double[w.Rows];
        for (var o = 0; o < w.Rows; o++)
        {
            double s = 0;
            var row = w.Row(o);
            for (var i = 0; i < x.Length; i++)
            {
                s += row[i] * x[i]; // same summation order as the scalar version
            }

            y[o] = s;
        }

        return y;
    }

    // backward for y = Wx:  dW += dy ⊗ x  and  dx = Wᵀ dy
    private static double[] LinearBackward(double[] x, Tensor2 w, double[] dy)
    {
        var dx = new double[x.Length];
        for (var o = 0; o < w.Rows; o++)
        {
            var g = dy[o];
            var row = w.Row(o);
            var grow = w.GradRow(o);
            for (var i = 0; i < x.Length; i++)
            {
                grow[i] += g * x[i];
                dx[i] += row[i] * g;
            }
        }

        return dx;
    }

    // rmsnorm forward: y_i = x_i * s ,  s = (mean(x²)+1e-5)^-½  — returns s for backward
    private static (double[] y, double s) RmsNorm(double[] x)
    {
        double ms = 0;
        for (var i = 0; i < x.Length; i++)
        {
            ms += x[i] * x[i];
        }

        ms /= x.Length;
        var s = Math.Pow(ms + 1e-5, -0.5);
        var y = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            y[i] = x[i] * s;
        }

        return (y, s);
    }

    // rmsnorm backward (analytical derivative):
    // dx_i = s·dy_i − (x_i·s³/n)·Σ_j dy_j·x_j
    private static double[] RmsNormBackward(double[] x, double s, double[] dy)
    {
        var n = x.Length;
        double dot = 0;
        for (var j = 0; j < n; j++) dot += dy[j] * x[j];
        var s3 = s * s * s;
        var dx = new double[n];
        for (var i = 0; i < n; i++)
        {
            dx[i] = s * dy[i] - x[i] * s3 * dot / n;
        }

        return dx;
    }

    // softmax (forward only — during training it is fused with cross-entropy: dlogits = p − onehot)
    private static double[] Softmax(double[] logits)
    {
        var maxVal = logits.Max();
        var e = new double[logits.Length];
        double total = 0;
        for (var i = 0; i < logits.Length; i++)
        {
            e[i] = Math.Exp(logits[i] - maxVal);
            total += e[i];
        }

        for (var i = 0; i < logits.Length; i++)
        {
            e[i] /= total;
        }

        return e;
    }

    // --- Per-position intermediate cache for backward (replaces a computational graph) ---
    private sealed class PosCache
    {
        public int TokenId, PosId;
        public double[] X0 = null!;
        public double S0; // embedding and first rmsnorm
        public double[] X1 = null!; // input to attention block (residual)
        public double[] X2 = null!;
        public double S1; // rmsnorm inside attention
        public double[] Q = null!; // query for this position
        public double[][] AttnW = null!; // attention weights per head
        public double[] XAttn = null!; // concatenated head outputs
        public double[] X4 = null!; // input to MLP block (residual)
        public double[] X5 = null!;
        public double S2; // rmsnorm inside MLP
        public double[] H1 = null!; // fc1 output (before ReLU)
        public double[] H2 = null!; // after ReLU
        public double[] X7 = null!; // final block output
        public double[] Probs = null!; // final softmax probabilities
    }

    private static readonly Dictionary<string, Tensor2> StateDict = new();

    public static void Main()
    {
        // Dataset — exactly like the scalar version (same RNG consumption order → identical initialization)
        const string inputFile = "input.txt";
        if (!File.Exists(inputFile))
        {
            using var http = new HttpClient();
            File.WriteAllText(inputFile, http.GetStringAsync(
                "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt").Result);
        }

        var docs = File.ReadAllLines(inputFile).Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
        for (var i = docs.Count - 1; i > 0; i--)
        {
            var j = Rng.Next(i + 1);
            (docs[i], docs[j]) = (docs[j], docs[i]);
        }

        Console.WriteLine($"num docs: {docs.Count}");

        var uchars = string.Concat(docs).Distinct().OrderBy(c => c).ToList();
        var BOS = uchars.Count;
        var vocabSize = uchars.Count + 1;
        var charToId = uchars.Select((c, i) => (c, i)).ToDictionary(t => t.c, t => t.i);
        Console.WriteLine($"vocab size: {vocabSize}");

        // Parameters — same construction order as the scalar version
        StateDict["wte"] = Matrix(vocabSize, NEmbd);
        StateDict["wpe"] = Matrix(BlockSize, NEmbd);
        StateDict["lm_head"] = Matrix(vocabSize, NEmbd);
        for (var i = 0; i < NLayer; i++)
        {
            StateDict[$"layer{i}.attn_wq"] = Matrix(NEmbd, NEmbd);
            StateDict[$"layer{i}.attn_wk"] = Matrix(NEmbd, NEmbd);
            StateDict[$"layer{i}.attn_wv"] = Matrix(NEmbd, NEmbd);
            StateDict[$"layer{i}.attn_wo"] = Matrix(NEmbd, NEmbd);
            StateDict[$"layer{i}.mlp_fc1"] = Matrix(4 * NEmbd, NEmbd);
            StateDict[$"layer{i}.mlp_fc2"] = Matrix(NEmbd, 4 * NEmbd);
        }

        var tensors = StateDict.Values.ToArray();
        var numParams = tensors.Sum(t => t.Data.Length);
        Console.WriteLine($"num params: {numParams}");

        // Adam — one m and v buffer per tensor
        double learningRate = 0.01, beta1 = 0.85, beta2 = 0.99, epsAdam = 1e-8;
        var mBuf = tensors.Select(t => new double[t.Data.Length]).ToArray();
        var vBuf = tensors.Select(t => new double[t.Data.Length]).ToArray();

        var wq = StateDict["layer0.attn_wq"];
        var wk = StateDict["layer0.attn_wk"];
        var wv = StateDict["layer0.attn_wv"];
        var wo = StateDict["layer0.attn_wo"];
        var fc1 = StateDict["layer0.mlp_fc1"];
        var fc2 = StateDict["layer0.mlp_fc2"];
        var wte = StateDict["wte"];
        var wpe = StateDict["wpe"];
        var lmHead = StateDict["lm_head"];
        var invSqrtHd = 1.0 / Math.Sqrt(HeadDim);

        const int numSteps = 1000;
        for (var step = 0; step < numSteps; step++)
        {
            var doc = docs[step % docs.Count];
            var tokens = new List<int> { BOS };
            tokens.AddRange(doc.Select(ch => charToId[ch]));
            tokens.Add(BOS);
            var n = Math.Min(BlockSize, tokens.Count - 1);

            // ---------------- Forward (saving intermediates for backward) ----------------
            var cache = new PosCache[n];
            var K = new double[n][]; // KV cache
            var V = new double[n][];
            double loss = 0;

            for (var t = 0; t < n; t++)
            {
                var c = new PosCache
                {
                    TokenId = tokens[t], PosId = t,
                    // embeddings
                    X0 = new double[NEmbd]
                };
                var te = wte.Row(c.TokenId);
                var pe = wpe.Row(t);
                for (var i = 0; i < NEmbd; i++)
                {
                    c.X0[i] = te[i] + pe[i];
                }

                (c.X1, c.S0) = RmsNorm(c.X0);

                // --- Attention block ---
                (c.X2, c.S1) = RmsNorm(c.X1);
                c.Q = Linear(c.X2, wq);
                K[t] = Linear(c.X2, wk);
                V[t] = Linear(c.X2, wv);

                c.XAttn = new double[NEmbd];
                c.AttnW = new double[NHead][];
                for (var h = 0; h < NHead; h++)
                {
                    var hs = h * HeadDim;
                    var logitsA = new double[t + 1];
                    for (var tp = 0; tp <= t; tp++)
                    {
                        double dot = 0;
                        for (var j = 0; j < HeadDim; j++) dot += c.Q[hs + j] * K[tp][hs + j];
                        logitsA[tp] = dot * invSqrtHd;
                    }

                    var aw = Softmax(logitsA);
                    c.AttnW[h] = aw;
                    for (var j = 0; j < HeadDim; j++)
                    {
                        double s = 0;
                        for (var tp = 0; tp <= t; tp++) s += aw[tp] * V[tp][hs + j];
                        c.XAttn[hs + j] = s;
                    }
                }

                var x3 = Linear(c.XAttn, wo);
                c.X4 = new double[NEmbd];
                for (var i = 0; i < NEmbd; i++) c.X4[i] = x3[i] + c.X1[i]; // residual

                // --- MLP block ---
                (c.X5, c.S2) = RmsNorm(c.X4);
                c.H1 = Linear(c.X5, fc1);
                c.H2 = new double[c.H1.Length];
                for (var i = 0; i < c.H1.Length; i++)
                {
                    c.H2[i] = Math.Max(0, c.H1[i]);
                }

                var x6 = Linear(c.H2, fc2);
                c.X7 = new double[NEmbd];
                for (var i = 0; i < NEmbd; i++)
                {
                    c.X7[i] = x6[i] + c.X4[i]; // residual
                }

                // logits and loss
                var logits = Linear(c.X7, lmHead);
                c.Probs = Softmax(logits);
                loss += -Math.Log(c.Probs[tokens[t + 1]]);
                cache[t] = c;
            }

            loss /= n;

            // ---------------- Backward (manual BPTT, reversed over positions) ----------------
            // Gradients for K and V at each position also arrive from later positions (causal attention),
            // so they accumulate in the reverse loop and are complete when we reach that position.
            var dK = new double[n][];
            var dV = new double[n][];
            for (var t = 0; t < n; t++)
            {
                dK[t] = new double[NEmbd];
                dV[t] = new double[NEmbd];
            }

            for (var t = n - 1; t >= 0; t--)
            {
                var c = cache[t];

                // cross-entropy+softmax derivative: dlogits = (p − onehot)/n
                var dlogits = new double[vocabSize];
                for (var i = 0; i < vocabSize; i++)
                {
                    dlogits[i] = c.Probs[i] / n;
                }

                dlogits[tokens[t + 1]] -= 1.0 / n;

                var dX7 = LinearBackward(c.X7, lmHead, dlogits);

                // MLP backward (residual: gradient flows to both branches)
                var dX6 = dX7; // fc2 branch
                var dH2 = LinearBackward(c.H2, fc2, dX6);
                var dH1 = new double[dH2.Length];
                for (var i = 0; i < dH1.Length; i++)
                {
                    dH1[i] = c.H1[i] > 0 ? dH2[i] : 0; // ReLU
                }

                var dX5 = LinearBackward(c.X5, fc1, dH1);
                var dX4 = RmsNormBackward(c.X4, c.S2, dX5);
                for (var i = 0; i < NEmbd; i++) dX4[i] += dX7[i]; // residual branch

                // Attention backward
                var dX3 = dX4;
                var dXAttn = LinearBackward(c.XAttn, wo, dX3);
                var dQ = new double[NEmbd];
                for (var h = 0; h < NHead; h++)
                {
                    var hs = h * HeadDim;
                    var aw = c.AttnW[h];
                    var T = t + 1;
                    // d(attn_weights) and d(V)
                    var da = new double[T];
                    for (var tp = 0; tp < T; tp++)
                    {
                        double s = 0;
                        for (var j = 0; j < HeadDim; j++)
                        {
                            s += dXAttn[hs + j] * V[tp][hs + j];
                            dV[tp][hs + j] += aw[tp] * dXAttn[hs + j];
                        }

                        da[tp] = s;
                    }

                    // softmax backward: dlogit_t' = a_t'·(da_t' − Σ a·da)
                    double dotAda = 0;
                    for (var tp = 0; tp < T; tp++) dotAda += aw[tp] * da[tp];
                    for (var tp = 0; tp < T; tp++)
                    {
                        var dl = aw[tp] * (da[tp] - dotAda) * invSqrtHd;
                        for (var j = 0; j < HeadDim; j++)
                        {
                            dQ[hs + j] += dl * K[tp][hs + j];
                            dK[tp][hs + j] += dl * c.Q[hs + j];
                        }
                    }
                }

                // dK[t] and dV[t] are now complete (all positions >= t have been processed)
                var dX2 = LinearBackward(c.X2, wq, dQ);
                var dX2k = LinearBackward(c.X2, wk, dK[t]);
                var dX2v = LinearBackward(c.X2, wv, dV[t]);
                for (var i = 0; i < NEmbd; i++)
                {
                    dX2[i] += dX2k[i] + dX2v[i];
                }

                var dX1 = RmsNormBackward(c.X1, c.S1, dX2);
                for (var i = 0; i < NEmbd; i++) dX1[i] += dX4[i]; // attention residual branch

                var dX0 = RmsNormBackward(c.X0, c.S0, dX1);
                var gte = wte.GradRow(c.TokenId);
                var gpe = wpe.GradRow(t);
                for (var i = 0; i < NEmbd; i++)
                {
                    gte[i] += dX0[i];
                    gpe[i] += dX0[i];
                }
            }

            // ---------------- Adam update (same formula as the scalar version) ----------------
            var lrT = learningRate * (1.0 - (double)step / numSteps);
            var bc1 = 1 - Math.Pow(beta1, step + 1);
            var bc2 = 1 - Math.Pow(beta2, step + 1);
            for (var ti = 0; ti < tensors.Length; ti++)
            {
                var tns = tensors[ti];
                var mm = mBuf[ti];
                var vv = vBuf[ti];
                for (var i = 0; i < tns.Data.Length; i++)
                {
                    var g = tns.Grad[i];
                    mm[i] = beta1 * mm[i] + (1 - beta1) * g;
                    vv[i] = beta2 * vv[i] + (1 - beta2) * g * g;
                    tns.Data[i] -= lrT * (mm[i] / bc1) / (Math.Sqrt(vv[i] / bc2) + epsAdam);
                    tns.Grad[i] = 0;
                }
            }

            if (step < 5 || (step + 1) % 100 == 0)
            {
                Console.WriteLine($"step {step + 1,4} / {numSteps,4} | loss {loss:F10}");
            }
        }

        // ---------------- Inference (pure forward pass, no gradients) ----------------
        var temperature = 0.5;
        Console.WriteLine("\n--- inference (new, hallucinated names) ---");
        for (var sampleIdx = 0; sampleIdx < 20; sampleIdx++)
        {
            var K = new List<double[]>();
            var V = new List<double[]>();
            var tokenId = BOS;
            var sample = new List<char>();
            for (var posId = 0; posId < BlockSize; posId++)
            {
                // forward one token (same architecture, no backward cache)
                var x0 = new double[NEmbd];
                var te = wte.Row(tokenId);
                var pe = wpe.Row(posId);
                for (var i = 0; i < NEmbd; i++)
                {
                    x0[i] = te[i] + pe[i];
                }
                var (x1, _) = RmsNorm(x0);
                var (x2, _) = RmsNorm(x1);
                var q = Linear(x2, wq);
                K.Add(Linear(x2, wk));
                V.Add(Linear(x2, wv));
                var xattn = new double[NEmbd];
                for (var h = 0; h < NHead; h++)
                {
                    var hs = h * HeadDim;
                    var la = new double[K.Count];
                    for (var tp = 0; tp < K.Count; tp++)
                    {
                        double dot = 0;
                        for (var j = 0; j < HeadDim; j++)
                        {
                            dot += q[hs + j] * K[tp][hs + j];
                        }

                        la[tp] = dot * invSqrtHd;
                    }

                    var aw = Softmax(la);
                    for (var j = 0; j < HeadDim; j++)
                    {
                        double s = 0;
                        for (var tp = 0; tp < K.Count; tp++)
                        {
                            s += aw[tp] * V[tp][hs + j];
                        }

                        xattn[hs + j] = s;
                    }
                }

                var x3 = Linear(xattn, wo);
                var x4 = new double[NEmbd];
                for (var i = 0; i < NEmbd; i++) x4[i] = x3[i] + x1[i];
                var (x5, _) = RmsNorm(x4);
                var h1 = Linear(x5, fc1);
                for (var i = 0; i < h1.Length; i++) h1[i] = Math.Max(0, h1[i]);
                var x6 = Linear(h1, fc2);
                var x7 = new double[NEmbd];
                for (var i = 0; i < NEmbd; i++) x7[i] = x6[i] + x4[i];
                var logits = Linear(x7, lmHead);
                for (var i = 0; i < logits.Length; i++) logits[i] /= temperature;
                var probs = Softmax(logits);

                var r = Rng.NextDouble() * probs.Sum();
                double cum = 0;
                tokenId = vocabSize - 1;
                for (var i = 0; i < probs.Length; i++)
                {
                    cum += probs[i];
                    if (!(r < cum))
                    {
                        continue;
                    }

                    tokenId = i;
                    break;
                }

                if (tokenId == BOS) break;
                sample.Add(uchars[tokenId]);
            }

            Console.WriteLine($"sample {sampleIdx + 1,2}: {string.Concat(sample)}");
        }
    }
}