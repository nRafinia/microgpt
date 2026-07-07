namespace MicroGptScalar;

// ----------------------------------------------------------------------------
// Autograd: Value class — scalar computational graph + chain rule
// Equivalent to the Python Value class with __slots__
// ----------------------------------------------------------------------------

public static class Program
{
    // --- Hyperparameters (identical to microgpt.py) ---
    private const int NLayer = 1; // number of transformer layers
    private const int NEmbd = 16; // embedding dimension
    private const int BlockSize = 16; // maximum sequence length
    private const int NHead = 4; // number of attention heads
    private const int HeadDim = NEmbd / NHead;

    private static readonly Dictionary<string, Value[][]> StateDict = new();
    private static readonly Random Rng = new Random(42); // Let there be order among chaos

    // Box-Muller method for random.gauss (normal distribution)
    private static double Gauss(double mu, double sigma)
    {
        var u1 = 1.0 - Rng.NextDouble();
        var u2 = Rng.NextDouble();
        return mu + sigma * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    // Equivalent to: matrix = lambda nout, nin, std=0.08: [[Value(gauss(0,std)) ...]]
    private static Value[][] Matrix(int nOut, int nIn, double std = 0.08)
    {
        var m = new Value[nOut][];
        for (var o = 0; o < nOut; o++)
        {
            m[o] = new Value[nIn];
            for (var i = 0; i < nIn; i++)
            {
                m[o][i] = new Value(Gauss(0, std));
            }
        }

        return m;
    }

    // linear: matrix-vector multiply — one dot product per row of w
    private static Value[] Linear(Value[] x, Value[][] w)
    {
        var outp = new Value[w.Length];
        for (var o = 0; o < w.Length; o++)
        {
            var s = new Value(0.0); // equivalent to Python's sum() starting from 0
            for (var i = 0; i < x.Length; i++)
            {
                s += w[o][i] * x[i];
            }

            outp[o] = s;
        }

        return outp;
    }

    // softmax: convert logits to a probability distribution (subtract max for numerical stability)
    private static Value[] Softmax(Value[] logits)
    {
        var maxVal = logits.Max(v => v.Data);
        var exps = new Value[logits.Length];
        for (var i = 0; i < logits.Length; i++)
        {
            exps[i] = (logits[i] - maxVal).Exp();
        }

        var total = new Value(0.0);
        foreach (var e in exps) total += e;
        var probs = new Value[logits.Length];
        for (var i = 0; i < logits.Length; i++)
        {
            probs[i] = exps[i] / total;
        }

        return probs;
    }

    // rmsnorm: normalize a vector to unit RMS
    private static Value[] RmsNorm(Value[] x)
    {
        var ms = new Value(0.0);
        foreach (var xi in x) ms += xi * xi;
        ms /= x.Length;
        var scale = (ms + 1e-5).Pow(-0.5);
        var outp = new Value[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            outp[i] = x[i] * scale;
        }

        return outp;
    }

    // ------------------------------------------------------------------------
    // Model architecture: GPT-2 with minor changes (rmsnorm instead of layernorm,
    // no bias, ReLU instead of GeLU). One token at one time position is processed + KV cache.
    // ------------------------------------------------------------------------
    private static Value[] Gpt(int tokenId, int posId, List<Value[]>[] keys, List<Value[]>[] values)
    {
        var tokEmb = StateDict["wte"][tokenId]; // token embedding
        var posEmb = StateDict["wpe"][posId]; // position embedding
        var x = new Value[NEmbd];
        for (var i = 0; i < NEmbd; i++)
        {
            x[i] = tokEmb[i] + posEmb[i]; // sum of token and position embeddings
        }

        x = RmsNorm(x); // Note: not redundant due to the residual path in backward pass

        for (var li = 0; li < NLayer; li++)
        {
            // 1) Multi-head Attention block
            var xResidual = x;
            x = RmsNorm(x);
            var q = Linear(x, StateDict[$"layer{li}.attn_wq"]);
            var k = Linear(x, StateDict[$"layer{li}.attn_wk"]);
            var v = Linear(x, StateDict[$"layer{li}.attn_wv"]);
            keys[li].Add(k);
            values[li].Add(v);

            var xAttn = new Value[NEmbd];
            for (var h = 0; h < NHead; h++)
            {
                var hs = h * HeadDim;
                var T = keys[li].Count; // number of positions available in cache

                // attn_logits[t] = (q_h · k_h[t]) / sqrt(head_dim)
                var attnLogits = new Value[T];
                for (var t = 0; t < T; t++)
                {
                    var dot = new Value(0.0);
                    for (var j = 0; j < HeadDim; j++)
                        dot += q[hs + j] * keys[li][t][hs + j];
                    attnLogits[t] = dot / Math.Sqrt(HeadDim);
                }

                var attnWeights = Softmax(attnLogits);

                // head_out[j] = Σ_t attn_weights[t] * v_h[t][j]
                for (var j = 0; j < HeadDim; j++)
                {
                    var s = new Value(0.0);
                    for (var t = 0; t < T; t++)
                        s += attnWeights[t] * values[li][t][hs + j];
                    xAttn[hs + j] = s; // equivalent to concat of head outputs
                }
            }

            x = Linear(xAttn, StateDict[$"layer{li}.attn_wo"]);
            for (var i = 0; i < NEmbd; i++)
                x[i] += xResidual[i]; // residual connection

            // 2) MLP block
            xResidual = x;
            x = RmsNorm(x);
            x = Linear(x, StateDict[$"layer{li}.mlp_fc1"]); // n_embd → 4*n_embd
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = x[i].Relu();
            }

            x = Linear(x, StateDict[$"layer{li}.mlp_fc2"]); // 4*n_embd → n_embd
            for (var i = 0; i < NEmbd; i++)
            {
                x[i] += xResidual[i]; // residual connection
            }
        }

        return Linear(x, StateDict["lm_head"]); // logits over the full vocabulary
    }

    public static void Main()
    {
        // --------------------------------------------------------------------
        // Dataset: list of documents (here: ~32k names, one per line)
        // --------------------------------------------------------------------
        const string inputFile = "input.txt";
        if (!File.Exists(inputFile))
        {
            using var http = new HttpClient();
            var data = http.GetStringAsync(
                "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt").Result;
            File.WriteAllText(inputFile, data);
        }

        var docs = File.ReadAllLines(inputFile)
            .Select(l => l.Trim())
            .Where(l => l.Length > 0)
            .ToList();
        // random.shuffle — Fisher-Yates algorithm
        for (var i = docs.Count - 1; i > 0; i--)
        {
            var j = Rng.Next(i + 1);
            (docs[i], docs[j]) = (docs[j], docs[i]);
        }

        Console.WriteLine($"num docs: {docs.Count}");

        // --------------------------------------------------------------------
        // Tokenizer: each unique character gets an id; + one special BOS token
        // --------------------------------------------------------------------
        var uchars = string.Concat(docs).Distinct().OrderBy(c => c).ToList();
        var BOS = uchars.Count;
        var vocabSize = uchars.Count + 1;
        var charToId = uchars.Select((c, i) => (c, i)).ToDictionary(t => t.c, t => t.i);
        Console.WriteLine($"vocab size: {vocabSize}");

        // --------------------------------------------------------------------
        // Parameters: model knowledge — weight matrices initialized with Gaussian noise
        // --------------------------------------------------------------------
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

        var params_ = StateDict.Values.SelectMany(m => m).SelectMany(r => r).ToArray();
        Console.WriteLine($"num params: {params_.Length}");

        // --------------------------------------------------------------------
        // Adam: optimizer with m (gradient mean) and v (squared gradient mean) buffers
        // --------------------------------------------------------------------
        double learningRate = 0.01, beta1 = 0.85, beta2 = 0.99, epsAdam = 1e-8;
        var m = new double[params_.Length];
        var v = new double[params_.Length];

        // --------------------------------------------------------------------
        // Training loop
        // --------------------------------------------------------------------
        const int numSteps = 1000;
        for (var step = 0; step < numSteps; step++)
        {
            // Pick one document, tokenize it and wrap with BOS on both sides: [BOS, e, m, m, a, BOS]
            var doc = docs[step % docs.Count];
            var tokens = new List<int> { BOS };
            tokens.AddRange(doc.Select(ch => charToId[ch]));
            tokens.Add(BOS);
            var n = Math.Min(BlockSize, tokens.Count - 1);

            // Forward: build the computational graph up to the loss
            var keys = Enumerable.Range(0, NLayer).Select(_ => new List<Value[]>()).ToArray();
            var values = Enumerable.Range(0, NLayer).Select(_ => new List<Value[]>()).ToArray();
            var losses = new List<Value>();
            for (var posId = 0; posId < n; posId++)
            {
                int tokenId = tokens[posId], targetId = tokens[posId + 1];
                var logits = Gpt(tokenId, posId, keys, values);
                var probs = Softmax(logits);
                var lossT = -probs[targetId].Log(); // cross-entropy: -log p(target)
                losses.Add(lossT);
            }

            var sumLoss = new Value(0.0);
            foreach (var l in losses)
            {
                sumLoss += l;
            }
            var loss = (1.0 / n) * sumLoss; // average loss over the document length

            // Backward: compute gradients for all parameters with a single call
            loss.Backward();

            // Adam update with linear learning rate decay
            var lrT = learningRate * (1.0 - (double)step / numSteps);
            for (var i = 0; i < params_.Length; i++)
            {
                var p = params_[i];
                m[i] = beta1 * m[i] + (1 - beta1) * p.Grad;
                v[i] = beta2 * v[i] + (1 - beta2) * p.Grad * p.Grad;
                var mHat = m[i] / (1 - Math.Pow(beta1, step + 1)); // bias correction
                var vHat = v[i] / (1 - Math.Pow(beta2, step + 1));
                p.Data -= lrT * mHat / (Math.Sqrt(vHat) + epsAdam);
                p.Grad = 0; // zero out gradient for the next step
            }

            if (step < 5 || (step + 1) % 100 == 0)
            {
                Console.WriteLine($"step {step + 1,4} / {numSteps,4} | loss {loss.Data:F10}");
            }
        }

        // --------------------------------------------------------------------
        // Inference: sample new names from the trained model
        // --------------------------------------------------------------------
        var temperature = 0.5; // in (0, 1] — controls output "creativity"
        Console.WriteLine("\n--- inference (new, hallucinated names) ---");
        for (var sampleIdx = 0; sampleIdx < 20; sampleIdx++)
        {
            var keys = Enumerable.Range(0, NLayer).Select(_ => new List<Value[]>()).ToArray();
            var values = Enumerable.Range(0, NLayer).Select(_ => new List<Value[]>()).ToArray();
            var tokenId = BOS;
            var sample = new List<char>();
            for (var posId = 0; posId < BlockSize; posId++)
            {
                var logits = Gpt(tokenId, posId, keys, values);
                var scaled = logits.Select(l => l / temperature).ToArray();
                var probs = Softmax(scaled);

                // random.choices with weights → sampling via cumulative sum
                var r = Rng.NextDouble() * probs.Sum(p => p.Data);
                double cum = 0;
                tokenId = vocabSize - 1;
                for (var i = 0; i < probs.Length; i++)
                {
                    cum += probs[i].Data;
                    if (!(r < cum))
                    {
                        continue;
                    }

                    tokenId = i;
                    break;
                }

                if (tokenId == BOS) break; // model signaled: "name is finished"
                sample.Add(uchars[tokenId]);
            }

            Console.WriteLine($"sample {sampleIdx + 1,2}: {string.Concat(sample)}");
        }
    }
}