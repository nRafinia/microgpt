namespace MicroGptScalar;

public sealed class Value(double data, Value[] children, double[] localGrads)
{
    public double Data = data;                    // scalar value (forward pass)
    public double Grad;                    // derivative of loss w.r.t. this node (backward pass)
    private readonly Value[] _children = children;    // children of this node in the computational graph
    private readonly double[] _localGrads = localGrads; // local derivative of this node w.r.t. each child

    private static readonly Value[] NoChildren = [];
    private static readonly double[] NoGrads = [];

    public Value(double data) : this(data, NoChildren, NoGrads)
    {
    }

    // Implicit conversion from double to Value (equivalent to __radd__ / __rmul__ etc. behavior in Python)
    public static implicit operator Value(double d) => new Value(d);

    // a + b  →  ∂/∂a = 1, ∂/∂b = 1
    public static Value operator +(Value a, Value b)
        => new Value(a.Data + b.Data, [a, b], [1.0, 1.0]);

    // a * b  →  ∂/∂a = b, ∂/∂b = a
    public static Value operator *(Value a, Value b)
        => new Value(a.Data * b.Data, [a, b], [b.Data, a.Data]);

    // a ** n (power with a constant)  →  ∂/∂a = n * a^(n-1)
    public Value Pow(double n)
        => new Value(Math.Pow(Data, n), [this], [n * Math.Pow(Data, n - 1)]);

    // log(a)  →  ∂/∂a = 1/a
    public Value Log()
        => new Value(Math.Log(Data), [this], [1.0 / Data]);

    // exp(a)  →  ∂/∂a = e^a
    public Value Exp()
    {
        var e = Math.Exp(Data);
        return new Value(e, [this], [e]);
    }

    // relu(a) = max(0, a)  →  ∂/∂a = 1[a > 0]
    public Value Relu()
        => new Value(Math.Max(0.0, Data), [this], [Data > 0 ? 1.0 : 0.0]);

    public static Value operator -(Value a) => a * -1.0;
    public static Value operator -(Value a, Value b) => a + (-b);
    public static Value operator /(Value a, Value b) => a * b.Pow(-1.0);

    // backward: traverse the graph in reverse topological order and apply the chain rule.
    // Note: the Python version builds the topology recursively;
    // here we implement the same DFS algorithm with an explicit stack (iterative) to
    // avoid StackOverflow on deep graphs. The output (topological order) is identical.
    public void Backward()
    {
        var topo = new List<Value>();
        var visited = new HashSet<Value>(ReferenceEqualityComparer.Instance);
        var stack = new Stack<(Value node, int childIdx)>();
        stack.Push((this, 0));
        visited.Add(this);

        while (stack.Count > 0)
        {
            var (node, idx) = stack.Pop();
            if (idx < node._children.Length)
            {
                stack.Push((node, idx + 1));
                var child = node._children[idx];
                if (visited.Add(child))
                    stack.Push((child, 0));
            }
            else
            {
                topo.Add(node); // all children have been processed
            }
        }

        Grad = 1.0; // ∂L/∂L = 1
        for (var i = topo.Count - 1; i >= 0; i--)
        {
            var v = topo[i];
            for (var c = 0; c < v._children.Length; c++)
                v._children[c].Grad += v._localGrads[c] * v.Grad; // +=: accumulate gradients from multiple paths
        }
    }
}