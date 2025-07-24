import Foundation

/// A type that represents uncertain data as a probability distribution
/// using sampling-based computation with conditional semantics.
///
/// `Uncertain` provides a way to work with probabilistic values
/// by representing them as sampling functions with a computation graph
/// for lazy evaluation and proper uncertainty-aware conditionals.
///
/// ## Example Usage
///
/// ```swift
/// let speed = Uncertain.normal(mean: 5.0, standardDeviation: 2.0)
///
/// // Paper-style conditionals: ask about evidence, not boolean facts
/// if (speed > 4.0).probability(exceeds: 0.9) {
///     print("90% confident you're going fast")
/// }
///
/// // Implicit conditional (equivalent to .probability(exceeds: 0.5))
/// if ~(speed > 4.0) {
///     print("More likely than not you're going fast")
/// }
/// ```
///
/// ## Performance Considerations
///
/// Uses Sequential Probability Ratio Test (SPRT) for efficient
/// hypothesis testing in conditionals. Sample counts are automatically
/// determined based on statistical significance rather than fixed numbers.
///
/// ## References
///
/// This library implements the approach described in:
///
/// James Bornholt, Todd Mytkowicz, and Kathryn S. McKinley.
/// "Uncertain<T>: A First-Order Type for Uncertain Data."
/// Architectural Support for Programming Languages and Operating Systems (ASPLOS), March 2014.
/// https://www.microsoft.com/en-us/research/publication/uncertaint-a-first-order-type-for-uncertain-data-2/
public struct Uncertain<T>: Sendable where T: Sendable {
    /// The sampling function that generates values from this distribution.
    public let sample: @Sendable () -> T

    /// The computation graph node for lazy evaluation
    internal let node: ComputationNode<T>

    /// Creates an uncertain value with the given sampling function.
    ///
    /// - Parameter sampler: A function that returns random samples from the distribution.
    public init(_ sample: @escaping @Sendable () -> T) {
        self.sample = sample
        self.node = .leaf(id: UUID(), sample: sample)
    }

    /// Internal initializer with computation node for building computation graphs
    init(sample: @escaping @Sendable () -> T, node: ComputationNode<T>) {
        self.sample = {
            var context = SampleContext()
            return node.evaluate(in: &context)
        }
        self.node = node
    }

    /// Transforms an uncertain value by applying a function to each sample.
    ///
    /// - Parameter transform: A function to apply to each sampled value.
    /// - Returns: A new uncertain value with the transformed distribution.
    public func map<U: Sendable>(_ transform: @escaping @Sendable (T) -> U) -> Uncertain<U> {
        Uncertain<U> { transform(self.sample()) }
    }

    /// Transforms an uncertain value by applying a function that returns another uncertain value.
    ///
    /// - Parameter transform: A function that takes a sample and returns an uncertain value.
    /// - Returns: A new uncertain value with the flattened distribution.
    public func flatMap<U: Sendable>(_ transform: @escaping @Sendable (T) -> Uncertain<U>)
        -> Uncertain<U>
    {
        Uncertain<U> { transform(self.sample()).sample() }
    }

    /// Filters samples using rejection sampling.
    ///
    /// Only samples that satisfy the predicate are accepted.
    /// This method will keep sampling until a valid sample is found,
    /// so ensure the predicate has a reasonable acceptance rate.
    ///
    /// - Parameter predicate: A function that returns `true` for accepted samples.
    /// - Returns: A new uncertain value with the filtered distribution.
    public func filter(_ predicate: @escaping @Sendable (T) -> Bool) -> Uncertain<T> {
        Uncertain<T> {
            var value: T
            repeat {
                value = self.sample()
            } while !predicate(value)
            return value
        }
    }
}

extension Uncertain: Sequence {
    /// Returns an iterator that produces infinite samples from this distribution.
    ///
    /// - Returns: An iterator that generates samples on demand.
    public func makeIterator() -> AnyIterator<T> {
        AnyIterator { self.sample() }
    }
}

// MARK: - Operators

extension Uncertain where T: Equatable {
    /// Returns uncertain boolean evidence that this value equals a given value
    /// This follows the paper's approach: comparisons return evidence, not boolean facts
    public static func == (lhs: Uncertain<T>, rhs: T) -> Uncertain<Bool> {
        let equalityNode = EqualityNode<T>.comparison(
            left: lhs.node,
            threshold: rhs,
            comparison: ==
        )

        return Uncertain<Bool>(
            sample: { equalityNode.evaluate() },
            node: ComputationNode<Bool>.leaf(id: UUID(), sample: { equalityNode.evaluate() })
        )
    }

    /// Returns uncertain boolean evidence that this value does not equal a given value
    public static func != (lhs: Uncertain<T>, rhs: T) -> Uncertain<Bool> {
        let equalityNode = EqualityNode<T>.comparison(
            left: lhs.node,
            threshold: rhs,
            comparison: !=
        )

        return Uncertain<Bool>(
            sample: { equalityNode.evaluate() },
            node: ComputationNode<Bool>.leaf(id: UUID(), sample: { equalityNode.evaluate() })
        )
    }

    /// Compares two uncertain values for equality
    public static func == (lhs: Uncertain<T>, rhs: Uncertain<T>) -> Uncertain<Bool> {
        return Uncertain<Bool> {
            lhs.sample() == rhs.sample()
        }
    }

    /// Compares two uncertain values for inequality
    public static func != (lhs: Uncertain<T>, rhs: Uncertain<T>) -> Uncertain<Bool> {
        return Uncertain<Bool> {
            lhs.sample() != rhs.sample()
        }
    }
}

extension Uncertain where T: Comparable {
    /// Returns uncertain boolean evidence that this value is greater than threshold
    /// This is the key insight from the paper: comparisons return evidence, not boolean facts
    public static func > (lhs: Uncertain<T>, rhs: T) -> Uncertain<Bool> {
        let comparisonNode = ComparisonNode<T>.comparison(
            left: lhs.node,
            threshold: rhs,
            comparison: >
        )

        return Uncertain<Bool>(
            sample: { comparisonNode.evaluate() },
            node: ComputationNode<Bool>.leaf(id: UUID(), sample: { comparisonNode.evaluate() })
        )
    }

    /// Returns uncertain boolean evidence that this value is less than threshold
    public static func < (lhs: Uncertain<T>, rhs: T) -> Uncertain<Bool> {
        let comparisonNode = ComparisonNode<T>.comparison(
            left: lhs.node,
            threshold: rhs,
            comparison: <
        )

        return Uncertain<Bool>(
            sample: { comparisonNode.evaluate() },
            node: ComputationNode<Bool>.leaf(id: UUID(), sample: { comparisonNode.evaluate() })
        )
    }

    /// Returns uncertain boolean evidence that this value is greater than or equal to threshold
    public static func >= (lhs: Uncertain<T>, rhs: T) -> Uncertain<Bool> {
        let comparisonNode = ComparisonNode<T>.comparison(
            left: lhs.node,
            threshold: rhs,
            comparison: >=
        )

        return Uncertain<Bool>(
            sample: { comparisonNode.evaluate() },
            node: ComputationNode<Bool>.leaf(id: UUID(), sample: { comparisonNode.evaluate() })
        )
    }

    /// Returns uncertain boolean evidence that this value is less than or equal to threshold
    public static func <= (lhs: Uncertain<T>, rhs: T) -> Uncertain<Bool> {
        let comparisonNode = ComparisonNode<T>.comparison(
            left: lhs.node,
            threshold: rhs,
            comparison: <=
        )

        return Uncertain<Bool>(
            sample: { comparisonNode.evaluate() },
            node: ComputationNode<Bool>.leaf(id: UUID(), sample: { comparisonNode.evaluate() })
        )
    }

    /// Compares two uncertain values
    public static func > (lhs: Uncertain<T>, rhs: Uncertain<T>) -> Uncertain<Bool> {
        return Uncertain<Bool> {
            lhs.sample() > rhs.sample()
        }
    }

    /// Compares two uncertain values
    public static func < (lhs: Uncertain<T>, rhs: Uncertain<T>) -> Uncertain<Bool> {
        return Uncertain<Bool> {
            lhs.sample() < rhs.sample()
        }
    }
}

extension Uncertain where T: Numeric {
    /// Adds two uncertain values - builds computation graph
    public static func + (lhs: Uncertain<T>, rhs: Uncertain<T>) -> Uncertain<T> {
        let newNode = ComputationNode<T>.binaryOp(
            left: lhs.node,
            right: rhs.node,
            operation: +
        )

        return Uncertain<T>(
            sample: { newNode.evaluate() },
            node: newNode
        )
    }

    /// Subtracts two uncertain values - builds computation graph
    public static func - (lhs: Uncertain<T>, rhs: Uncertain<T>) -> Uncertain<T> {
        let newNode = ComputationNode<T>.binaryOp(
            left: lhs.node,
            right: rhs.node,
            operation: -
        )

        return Uncertain<T>(
            sample: { newNode.evaluate() },
            node: newNode
        )
    }

    /// Multiplies two uncertain values - builds computation graph
    public static func * (lhs: Uncertain<T>, rhs: Uncertain<T>) -> Uncertain<T> {
        let newNode = ComputationNode<T>.binaryOp(
            left: lhs.node,
            right: rhs.node,
            operation: *
        )

        return Uncertain<T>(
            sample: { newNode.evaluate() },
            node: newNode
        )
    }

    /// Adds a constant to an uncertain value
    public static func + (lhs: Uncertain<T>, rhs: T) -> Uncertain<T> {
        return lhs + Uncertain<T> { rhs }
    }

    /// Adds an uncertain value to a constant (commutative)
    public static func + (lhs: T, rhs: Uncertain<T>) -> Uncertain<T> {
        return Uncertain<T> { lhs } + rhs
    }

    /// Subtracts a constant from an uncertain value
    public static func - (lhs: Uncertain<T>, rhs: T) -> Uncertain<T> {
        return lhs - Uncertain<T> { rhs }
    }

    /// Subtracts an uncertain value from a constant
    public static func - (lhs: T, rhs: Uncertain<T>) -> Uncertain<T> {
        return Uncertain<T> { lhs } - rhs
    }

    /// Multiplies an uncertain value by a constant
    public static func * (lhs: Uncertain<T>, rhs: T) -> Uncertain<T> {
        return lhs * Uncertain<T> { rhs }
    }

    /// Multiplies a constant by an uncertain value (commutative)
    public static func * (lhs: T, rhs: Uncertain<T>) -> Uncertain<T> {
        return Uncertain<T> { lhs } * rhs
    }
}

extension Uncertain where T: FixedWidthInteger {
    /// Divides two uncertain integer values.
    public static func / (lhs: Uncertain<T>, rhs: Uncertain<T>) -> Uncertain<T> {
        return Uncertain<T> {
            let divisor = rhs.sample()
            precondition(divisor != 0, "Division by zero in Uncertain integer division")
            return lhs.sample() / divisor
        }
    }

    /// Divides an uncertain integer value by a constant.
    public static func / (lhs: Uncertain<T>, rhs: T) -> Uncertain<T> {
        precondition(rhs != 0, "Division by zero in Uncertain integer division")
        return Uncertain<T> {
            lhs.sample() / rhs
        }
    }

    /// Divides a constant by an uncertain integer value.
    public static func / (lhs: T, rhs: Uncertain<T>) -> Uncertain<T> {
        return Uncertain<T> {
            let divisor = rhs.sample()
            precondition(divisor != 0, "Division by zero in Uncertain integer division")
            return lhs / divisor
        }
    }
}

extension Uncertain where T: BinaryFloatingPoint {
    /// Divides two uncertain values - builds computation graph
    public static func / (lhs: Uncertain<T>, rhs: Uncertain<T>) -> Uncertain<T> {
        let newNode = ComputationNode<T>.binaryOp(
            left: lhs.node,
            right: rhs.node,
            operation: /
        )

        return Uncertain<T>(
            sample: { newNode.evaluate() },
            node: newNode
        )
    }

    /// Divides an uncertain value by a constant
    public static func / (lhs: Uncertain<T>, rhs: T) -> Uncertain<T> {
        return lhs / Uncertain<T> { rhs }
    }

    /// Divides a constant by an uncertain value
    public static func / (lhs: T, rhs: Uncertain<T>) -> Uncertain<T> {
        return Uncertain<T> { lhs } / rhs
    }
}

/// Prefix operator for implicit conditional (equivalent to .probability(exceeds: 0.5))
prefix operator ~

extension Uncertain where T == Bool {
    /// Prefix operator for implicit conditional
    public static prefix func ~ (operand: Uncertain<Bool>) -> Bool {
        return operand.implicitConditional()
    }

    /// Logical NOT operator for uncertain boolean values
    public static prefix func ! (operand: Uncertain<Bool>) -> Uncertain<Bool> {
        return Uncertain<Bool> {
            var context = SampleContext()
            return !operand.node.evaluate(in: &context)
        }
    }

    /// Logical AND operator for uncertain boolean values
    public static func && (lhs: Uncertain<Bool>, rhs: Uncertain<Bool>) -> Uncertain<Bool> {
        return Uncertain<Bool> {
            var context = SampleContext()
            return lhs.node.evaluate(in: &context) && rhs.node.evaluate(in: &context)
        }
    }

    /// Logical OR operator for uncertain boolean values
    public static func || (lhs: Uncertain<Bool>, rhs: Uncertain<Bool>) -> Uncertain<Bool> {
        return Uncertain<Bool> {
            var context = SampleContext()
            return lhs.node.evaluate(in: &context) || rhs.node.evaluate(in: &context)
        }
    }
}

// MARK: - Distributions

extension Uncertain {
    /// Creates a point-mass distribution (certain value) as per the paper.
    ///
    /// - Parameter value: The certain value to always return.
    /// - Returns: A new uncertain value that always returns the same value.
    public static func point(_ value: T) -> Uncertain<T> {
        return Uncertain<T> { value }
    }

    /// Creates a mixture of distributions with optional weights.
    ///
    /// - Parameters:
    ///   - components: An array of distributions to mix.
    ///   - weights: An optional array of weights corresponding to each distribution.
    ///              If `nil`, uses uniform weights.
    /// - Returns: A new uncertain value representing the mixture distribution.
    /// - Precondition: If weights are provided, `components.count == weights.count`.
    public static func mixture(of components: [Uncertain<T>], weights: [Double]? = nil)
        -> Uncertain<T>
    {
        precondition(!components.isEmpty, "At least one component required")
        if components.count == 1 {
            return components[0]
        }
        let w: [Double]
        if let weights = weights {
            precondition(
                components.count == weights.count, "Weights count must match components count")
            w = weights
        } else {
            w = Array(repeating: 1.0, count: components.count)
        }
        let total = w.reduce(0, +)
        let normalized = w.map { $0 / total }
        let cumulative = normalized.reduce(into: [Double]()) { acc, weight in
            acc.append((acc.last ?? 0) + weight)
        }
        return Uncertain<T> {
            let r = Double.random(in: 0...1)
            let idx = cumulative.firstIndex(where: { r <= $0 }) ?? (components.count - 1)
            return components[idx].sample()
        }
    }

    /// Creates an empirical distribution from observed data.
    ///
    /// - Parameter data: An array of observed values.
    /// - Returns: A new uncertain value that randomly selects from the provided data, or nil if data is empty.
    public static func empirical(_ data: [T]) -> Uncertain<T>? {
        guard !data.isEmpty else { return nil }
        return Uncertain<T> { data.randomElement()! }
    }
}

extension Uncertain where T: Hashable {
    /// Creates a categorical distribution from value-probability pairs.
    ///
    /// - Parameter probabilities: A dictionary mapping values to their probabilities.
    /// - Returns: A new uncertain value representing the categorical distribution, or nil if input is empty.
    public static func categorical(_ probabilities: [T: Double]) -> Uncertain<T>? {
        guard !probabilities.isEmpty else { return nil }
        let total = probabilities.values.reduce(0, +)
        let normalized = probabilities.mapValues { $0 / total }
        let sorted = normalized.sorted { $0.value < $1.value }
        let cumulative = sorted.reduce(into: [(T, Double)]()) { acc, pair in
            let sum = (acc.last?.1 ?? 0) + pair.value
            acc.append((pair.key, sum))
        }
        return Uncertain<T> {
            let r = Double.random(in: 0...1)
            // Defensive: fallback to last element if none found (should not happen if cumulative is non-empty)
            return cumulative.first(where: { r <= $0.1 })?.0 ?? cumulative.last!.0
        }
    }
}

extension Uncertain where T: FixedWidthInteger {

    /// Creates a binomial distribution for any integer type.
    ///
    /// - Parameters:
    ///   - trials: The number of trials.
    ///   - probability: The probability of success on each trial.
    /// - Returns: A new uncertain value with a binomial distribution.
    public static func binomial(trials: T, probability: Double) -> Uncertain<T> {
        Uncertain<T> {
            var count: T = 0
            for _ in 0..<trials {
                if Double.random(in: 0...1) < probability {
                    count += 1
                }
            }
            return count
        }
    }

    /// Creates a Poisson distribution for any integer type.
    ///
    /// - Parameter lambda: The rate parameter.
    /// - Returns: A new uncertain value with a Poisson distribution.
    public static func poisson(lambda: Double) -> Uncertain<T> {
        Uncertain<T> {
            let L = exp(-lambda)
            var k: T = 0
            var p = 1.0
            repeat {
                k += 1
                p *= Double.random(in: 0...1)
            } while p > L
            return k - 1
        }
    }
}

extension Uncertain where T == Double {
    /// Creates a normal (Gaussian) distribution.
    ///
    /// - Parameters:
    ///   - mean: The mean of the distribution.
    ///   - standardDeviation: The standard deviation of the distribution.
    /// - Returns: A new uncertain value with a normal distribution.
    public static func normal(
        mean: Double,
        standardDeviation: Double
    ) -> Uncertain<Double> {
        return Uncertain<Double> {
            // Box-Muller transform for normal distribution
            let u1 = Double.random(in: 0.001...0.999)  // Avoid exactly 0 or 1
            let u2 = Double.random(in: 0.001...0.999)
            let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * Double.pi * u2)
            return mean + standardDeviation * z0
        }
    }

    /// Creates a uniform distribution.
    ///
    /// - Parameters:
    ///   - min: The minimum value.
    ///   - max: The maximum value.
    /// - Returns: A new uncertain value with a uniform distribution.
    public static func uniform(min: Double, max: Double) -> Uncertain<Double> {
        return Uncertain<Double> {
            Double.random(in: min...max)
        }
    }

    /// Creates an exponential distribution.
    ///
    /// - Parameter rate: The rate parameter (lambda).
    /// - Returns: A new uncertain value with an exponential distribution.
    public static func exponential(rate: Double) -> Uncertain<Double> {
        return Uncertain<Double> {
            -log(Double.random(in: 0.0...1.0)) / rate
        }
    }

    /// Creates a Bernoulli distribution.
    ///
    /// - Parameter probability: The probability of success.
    /// - Returns: A new uncertain boolean value.
    public static func bernoulli(probability: Double) -> Uncertain<Bool> {
        Uncertain<Bool> { Double.random(in: 0...1) < probability }
    }

    /// Creates a Kumaraswamy distribution.
    ///
    /// - Parameters:
    ///   - a: The first shape parameter (must be > 0).
    ///   - b: The second shape parameter (must be > 0).
    /// - Returns: A new uncertain value with a Kumaraswamy distribution in range [0, 1].
    public static func kumaraswamy(a: Double, b: Double) -> Uncertain<Double> {
        precondition(a > 0 && b > 0, "Kumaraswamy distribution parameters must be positive")

        // Cache reciprocals to avoid repeated division
        let reciprocalA = 1.0 / a
        let reciprocalB = 1.0 / b

        return Uncertain<Double> {
            // Generate Kumaraswamy using inverse transform sampling
            let u = Double.random(in: Double.ulpOfOne..<1.0)  // Avoid exactly 0 or 1
            return pow(1.0 - pow(1.0 - u, reciprocalB), reciprocalA)
        }
    }

    /// Creates a Rayleigh distribution.
    ///
    /// The Rayleigh distribution models the magnitude of a 2D vector whose components
    /// are normally distributed. It's commonly used for modeling distances from a center point.
    ///
    /// - Parameter scale: The scale parameter (must be > 0).
    /// - Returns: A new uncertain value with a Rayleigh distribution.
    public static func rayleigh(scale: Double) -> Uncertain<Double> {
        precondition(scale > 0, "Rayleigh distribution scale parameter must be positive")

        return Uncertain<Double> {
            // Generate Rayleigh using inverse transform sampling
            // F^(-1)(u) = scale * sqrt(-2 * ln(1-u))
            let u = Double.random(in: Double.ulpOfOne..<1.0)  // Avoid exactly 0 or 1
            return scale * sqrt(-2.0 * log(1.0 - u))
        }
    }

    /// Estimates the log-likelihood of a value using kernel density estimation.
    ///
    /// - Parameters:
    ///   - value: The value to evaluate.
    ///   - sampleCount: The number of samples to use for estimation.
    ///   - bandwidth: The bandwidth parameter for the kernel.
    /// - Returns: The estimated log-likelihood.
    public func logLikelihood(_ value: Double, sampleCount: Int = 1000, bandwidth: Double = 1.0)
        -> Double
    {
        let samples = self.prefix(sampleCount)
        let kernel = { (x: Double, xi: Double) -> Double in
            exp(-0.5 * pow((x - xi) / bandwidth, 2)) / (bandwidth * sqrt(2 * .pi))
        }
        let density = samples.map { kernel(value, $0) }.reduce(0, +) / Double(sampleCount)
        return log(density)
    }
}

// MARK: - Statistics

extension Uncertain where T: Hashable {
    /// Returns the most frequently occurring value in the distribution.
    ///
    /// - Parameter sampleCount: The number of samples to use for estimation.
    /// - Returns: The mode (most frequent value), or `nil` if no samples are available.
    public func mode(sampleCount: Int = 1000) -> T? {
        let samples = Array(self.prefix(sampleCount))
        guard !samples.isEmpty else { return nil }
        let counts = Dictionary(grouping: samples, by: { $0 }).mapValues { $0.count }
        return counts.max(by: { $0.value < $1.value })?.key
    }

    /// Returns a histogram showing the frequency of each value.
    ///
    /// - Parameter sampleCount: The number of samples to use for estimation.
    /// - Returns: A dictionary mapping values to their occurrence counts.
    /// - Note: Performance is O(n) with memory usage O(`sampleCount` + unique values).
    /// For high-cardinality data, consider binning approaches.
    public func histogram(sampleCount: Int = 1000) -> [T: Int] {
        let samples = self.prefix(sampleCount)
        return Dictionary(grouping: samples, by: { $0 }).mapValues { $0.count }
    }

    /// Calculates the empirical entropy of the distribution.
    ///
    /// - Parameter sampleCount: The number of samples to use for estimation.
    /// - Returns: The entropy in bits.
    public func entropy(sampleCount: Int = 1000) -> Double {
        let samples = Array(self.prefix(sampleCount))
        let counts = Dictionary(grouping: samples, by: { $0 }).mapValues { $0.count }
        let total = Double(samples.count)
        return counts.values.reduce(0.0) { acc, count in
            let p = Double(count) / total
            return acc - (p > 0 ? p * log2(p) : 0)
        }
    }
}

extension Uncertain where T: FixedWidthInteger {
    /// Calculates the expected value (mean) of the integer distribution as Double.
    public func expectedValue(sampleCount: Int = 1000) -> Double {
        let samples = self.prefix(sampleCount)
        let sum = samples.reduce(0) { $0 + Int($1) }
        return Double(sum) / Double(sampleCount)
    }

    /// Calculates the standard deviation of the integer distribution as Double.
    public func standardDeviation(sampleCount: Int = 1000) -> Double {
        let samples = self.prefix(sampleCount)
        let mean = self.expectedValue(sampleCount: sampleCount)
        let variance =
            samples.reduce(0.0) { sum, sample in
                let diff = Double(sample) - mean
                return sum + diff * diff
            } / Double(sampleCount)
        return sqrt(variance)
    }
}

extension Uncertain where T: BinaryFloatingPoint {
    /// Calculates the expected value (mean) of the distribution.
    ///
    /// - Parameter sampleCount: The number of samples to use for estimation.
    /// - Returns: The expected value.
    public func expectedValue(sampleCount: Int = 1000) -> T {
        let samples = self.prefix(sampleCount)
        return samples.reduce(T.zero, +) / T(sampleCount)
    }

    /// Calculates the skewness of the distribution.
    ///
    /// - Parameter sampleCount: The number of samples to use for estimation.
    /// - Returns: The skewness value.
    public func skewness(sampleCount: Int = 1000) -> Double {
        let samples = self.prefix(sampleCount)
        let mean = samples.reduce(T.zero, +) / T(sampleCount)
        let variance =
            samples.reduce(T.zero) { sum, sample in
                let diff = sample - mean
                return sum + diff * diff
            } / T(sampleCount)
        let std = Double(variance).squareRoot()
        let n = Double(sampleCount)
        let skew = samples.reduce(0.0) { $0 + pow(Double($1 - mean), 3) }
        return skew / n / pow(std, 3)
    }

    /// Calculates the kurtosis of the distribution.
    ///
    /// - Parameter sampleCount: The number of samples to use for estimation.
    /// - Returns: The excess kurtosis value.
    public func kurtosis(sampleCount: Int = 1000) -> Double {
        let samples = self.prefix(sampleCount)
        let mean = samples.reduce(T.zero, +) / T(sampleCount)
        let variance =
            samples.reduce(T.zero) { sum, sample in
                let diff = sample - mean
                return sum + diff * diff
            } / T(sampleCount)
        let std = Double(variance).squareRoot()
        let n = Double(sampleCount)
        let kurt = samples.reduce(0.0) { $0 + pow(Double($1 - mean), 4) }
        return kurt / n / pow(std, 4) - 3.0
    }
}

extension Uncertain where T: Comparable & BinaryFloatingPoint {
    /// Calculates the confidence interval for the distribution.
    ///
    /// - Parameters:
    ///   - confidence: The confidence level (e.g., 0.95 for 95% CI).
    ///   - sampleCount: The number of samples to use for estimation.
    /// - Returns: A tuple containing the lower and upper bounds of the confidence interval.
    /// - Note: Performance is O(n log n) due to sorting.
    ///         Memory usage is O(`sampleCount`).
    ///         For large sample counts, consider using quantile-based approaches.
    public func confidenceInterval(_ confidence: Double = 0.95, sampleCount: Int = 1000) -> (
        lower: T, upper: T
    ) {
        let samples = Array(self.prefix(sampleCount)).sorted()
        let alpha = 1.0 - confidence
        let lowerIndex = Int(alpha / 2.0 * Double(samples.count))
        let upperIndex = Int((1.0 - alpha / 2.0) * Double(samples.count)) - 1

        return (
            lower: samples[Swift.max(0, Swift.min(lowerIndex, samples.count - 1))],
            upper: samples[Swift.max(0, Swift.min(upperIndex, samples.count - 1))]
        )
    }

    /// Calculates the standard deviation of the distribution.
    ///
    /// - Parameter sampleCount: The number of samples to use for estimation.
    /// - Returns: The standard deviation.
    /// - Note: Performance is O(n) with two passes through the data.
    ///         Memory usage is O(`sampleCount`).
    public func standardDeviation(sampleCount: Int = 1000) -> Double {
        let samples = self.prefix(sampleCount)
        let mean = samples.reduce(T.zero, +) / T(sampleCount)
        let variance =
            samples.reduce(T.zero) { sum, sample in
                let diff = sample - mean
                return sum + diff * diff
            } / T(sampleCount)
        return Double(variance).squareRoot()
    }

    /// Estimates the cumulative distribution function (CDF) at a given value.
    ///
    /// - Parameters:
    ///   - value: The value at which to evaluate the CDF.
    ///   - sampleCount: The number of samples to use for estimation.
    /// - Returns: The CDF value (probability that a sample is ≤ value).
    public func cdf(at value: T, sampleCount: Int = 1000) -> Double {
        let samples = self.prefix(sampleCount)
        let successes = samples.filter { $0 <= value }.count
        return Double(successes) / Double(sampleCount)
    }
}

/// QuickSelect algorithm for finding the k-th smallest element in O(n) average time
private func quickSelect<T: Comparable>(_ array: inout [T], k: Int) -> T {
    precondition(k >= 0 && k < array.count, "Index out of bounds")

    var left = 0
    var right = array.count - 1

    while left < right {
        let pivotIndex = partition(&array, left: left, right: right)
        if pivotIndex == k {
            return array[k]
        } else if pivotIndex < k {
            left = pivotIndex + 1
        } else {
            right = pivotIndex - 1
        }
    }

    return array[k]
}

/// Partition function for QuickSelect
private func partition<T: Comparable>(_ array: inout [T], left: Int, right: Int) -> Int {
    let pivot = array[right]
    var i = left

    for j in left..<right {
        if array[j] <= pivot {
            array.swapAt(i, j)
            i += 1
        }
    }

    array.swapAt(i, right)
    return i
}

// MARK: - Hypothesis Testing

extension Uncertain where T == Bool {
    /// Swift-conventional explicit conditional operator using hypothesis testing
    /// Uses Sequential Probability Ratio Test (SPRT) for efficiency
    public func probability(
        exceeds threshold: Double,
        confidenceLevel: Double = 0.95,
        maxSamples: Int = 10000
    ) -> Bool {
        let result = evaluateHypothesis(
            threshold: threshold,
            confidenceLevel: confidenceLevel,
            maxSamples: maxSamples
        )
        return result.decision
    }

    /// Implicit conditional operator (equivalent to .probability(exceeds: 0.5))
    public func implicitConditional(confidenceLevel: Double = 0.95) -> Bool {
        return probability(exceeds: 0.5, confidenceLevel: confidenceLevel)
    }

    /// Performs SPRT-inspired hypothesis testing with configurable parameters
    /// H0: P(true) <= threshold vs H1: P(true) > threshold
    /// Returns a tuple containing the decision, as well as the probability, confidence level, and samples used
    public func evaluateHypothesis(
        threshold: Double,
        confidenceLevel: Double,
        maxSamples: Int,
        epsilon: Double = 0.05,
        alpha: Double? = nil,
        beta: Double? = nil,
        batchSize: Int = 10
    ) -> (
        decision: Bool,
        probability: Double,
        confidenceLevel: Double,
        samplesUsed: Int
    ) {
        // Set alpha and beta for the test
        let alphaError = alpha ?? (1.0 - confidenceLevel)  // Type I error (false positive rate)
        let betaError = beta ?? alphaError  // Type II error (false negative rate)

        // Calculate SPRT boundaries
        let A = log(betaError / (1.0 - alphaError))
        let B = log((1.0 - betaError) / alphaError)

        // Set indifference region (epsilon)
        let p0 = Swift.max(0.0, Swift.min(1.0, threshold - epsilon))  // Null hypothesis
        let p1 = Swift.max(0.0, Swift.min(1.0, threshold + epsilon))  // Alternative hypothesis

        var successes = 0
        var samples = 0

        while samples < maxSamples {
            // Batch sampling as per paper
            var batchSuccesses = 0
            let currentBatchSize = Swift.min(batchSize, maxSamples - samples)

            for _ in 0..<currentBatchSize {
                if self.sample() {
                    batchSuccesses += 1
                }
            }

            successes += batchSuccesses
            samples += currentBatchSize

            // Compute log-likelihood ratio (LLR)
            let n = samples
            let x = successes
            // Avoid log(0) by clamping probabilities
            let p0c = Swift.max(1e-10, Swift.min(1.0 - 1e-10, p0))
            let p1c = Swift.max(1e-10, Swift.min(1.0 - 1e-10, p1))
            let llr = Double(x) * log(p1c / p0c) + Double(n - x) * log((1.0 - p1c) / (1.0 - p0c))

            if llr <= A {
                // Accept H0: P(true) <= threshold
                let p = Double(successes) / Double(samples)
                return (
                    decision: false,
                    probability: p,
                    confidenceLevel: confidenceLevel,
                    samplesUsed: samples
                )
            } else if llr >= B {
                // Accept H1: P(true) > threshold
                let p = Double(successes) / Double(samples)
                return (
                    decision: true,
                    probability: p,
                    confidenceLevel: confidenceLevel,
                    samplesUsed: samples
                )
            }
        }

        // Fallback decision based on observed probability
        let finalP = Double(successes) / Double(samples)
        return (
            decision: finalP > threshold,
            probability: finalP,
            confidenceLevel: confidenceLevel,
            samplesUsed: samples
        )
    }
}

// MARK: - Core Location

#if canImport(CoreLocation)
    import CoreLocation

    extension Uncertain where T == CLLocation {
        /// Creates an uncertain location from a CLLocation.
        ///
        /// Models GPS uncertainty using a Rayleigh distribution
        /// for radial distance from the true location,
        /// combined with a uniform angular distribution.
        ///
        /// - Parameter location: The CLLocation with accuracy information.
        /// - Returns: A new uncertain location value.
        public static func from(_ location: CLLocation) -> Uncertain<CLLocation> {
            return Uncertain<CLLocation> {
                let horizontalAccuracy = location.horizontalAccuracy
                let verticalAccuracy = location.verticalAccuracy

                // If accuracy is negative, the reading is invalid
                guard horizontalAccuracy >= 0 else {
                    return location
                }

                // Model horizontal uncertainty using Rayleigh distribution for radial distance
                let earthRadius = 6_371_000.0  // meters
                let lat = location.coordinate.latitude * .pi / 180

                // Generate radial distance using Rayleigh distribution
                // horizontalAccuracy represents the standard deviation of GPS error
                let radialDistance = Uncertain<Double>.rayleigh(
                    scale: horizontalAccuracy
                ).sample()

                // Generate random direction (angle) uniformly
                let angle = Double.random(in: 0...(2 * .pi))

                // Convert polar coordinates to Cartesian offsets
                let northOffset = radialDistance * cos(angle)
                let eastOffset = radialDistance * sin(angle)

                // Convert meter offsets to lat/lon
                let latOffset = northOffset / earthRadius * 180 / .pi
                let lonOffset = eastOffset / (earthRadius * cos(lat)) * 180 / .pi

                let uncertainCoordinate = CLLocationCoordinate2D(
                    latitude: location.coordinate.latitude + latOffset,
                    longitude: location.coordinate.longitude + lonOffset
                )

                // Model altitude uncertainty if available
                let uncertainAltitude: Double
                if verticalAccuracy >= 0 {
                    uncertainAltitude =
                        location.altitude
                        + Uncertain<Double>.normal(mean: 0, standardDeviation: verticalAccuracy)
                        .sample()
                } else {
                    uncertainAltitude = location.altitude
                }

                return CLLocation(
                    coordinate: uncertainCoordinate,
                    altitude: uncertainAltitude,
                    horizontalAccuracy: horizontalAccuracy,
                    verticalAccuracy: verticalAccuracy,
                    timestamp: location.timestamp
                )
            }
        }

        /// Creates an uncertain location with specified uncertainties.
        ///
        /// - Parameters:
        ///   - coordinate: The GPS coordinate.
        ///   - horizontalAccuracy: The horizontal accuracy in meters.
        ///   - verticalAccuracy: The vertical accuracy in meters.
        ///   - altitude: The altitude in meters.
        /// - Returns: A new uncertain location value.
        public static func location(
            coordinate: CLLocationCoordinate2D,
            horizontalAccuracy: CLLocationAccuracy,
            verticalAccuracy: CLLocationAccuracy = -1,
            altitude: CLLocationDistance = 0
        ) -> Uncertain<CLLocation> {
            let baseLocation = CLLocation(
                coordinate: coordinate,
                altitude: altitude,
                horizontalAccuracy: horizontalAccuracy,
                verticalAccuracy: verticalAccuracy,
                timestamp: Date()
            )
            return from(baseLocation)
        }

        /// Calculates the distance between two uncertain locations.
        ///
        /// - Parameters:
        ///   - from: The starting location.
        ///   - to: The ending location.
        /// - Returns: An uncertain distance value in meters.
        public static func distance(
            from: Uncertain<CLLocation>,
            to: Uncertain<CLLocation>
        ) -> Uncertain<CLLocationDistance> {
            return Uncertain<CLLocationDistance> {
                let loc1 = from.sample()
                let loc2 = to.sample()
                return loc1.distance(from: loc2)
            }
        }

        /// Calculates the bearing between two uncertain locations.
        ///
        /// - Parameters:
        ///   - from: The starting location.
        ///   - to: The ending location.
        /// - Returns: An uncertain bearing value in degrees (0-360).
        public static func bearing(
            from: Uncertain<CLLocation>,
            to: Uncertain<CLLocation>
        ) -> Uncertain<Double> {
            return Uncertain<Double> {
                let loc1 = from.sample()
                let loc2 = to.sample()

                let lat1 = loc1.coordinate.latitude * .pi / 180
                let lat2 = loc2.coordinate.latitude * .pi / 180
                let deltaLon = (loc2.coordinate.longitude - loc1.coordinate.longitude) * .pi / 180

                let x = sin(deltaLon) * cos(lat2)
                let y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(deltaLon)
                let bearing = atan2(x, y)

                // Convert to degrees and normalize to 0-360
                var degrees = bearing * 180 / .pi
                if degrees < 0 {
                    degrees += 360
                }
                return degrees
            }
        }

        /// Calculates the distance from this location to another location.
        ///
        /// - Parameter location: The destination location.
        /// - Returns: An uncertain distance value in meters.
        public func distance(to location: Uncertain<CLLocation>) -> Uncertain<CLLocationDistance> {
            return Self.distance(from: self, to: location)
        }

        /// Calculates the distance from this location to a specific location.
        ///
        /// - Parameter location: The destination location.
        /// - Returns: An uncertain distance value in meters.
        public func distance(to location: CLLocation) -> Uncertain<CLLocationDistance> {
            return Self.distance(from: self, to: Uncertain<CLLocation>.from(location))
        }

        /// Calculates the bearing from this location to another location.
        ///
        /// - Parameter location: The destination location.
        /// - Returns: An uncertain bearing value in degrees (0-360).
        public func bearing(to location: Uncertain<CLLocation>) -> Uncertain<Double> {
            return Self.bearing(from: self, to: location)
        }

        /// Calculates the bearing from this location to a specific location.
        ///
        /// - Parameter location: The destination location.
        /// - Returns: An uncertain bearing value in degrees (0-360).
        public func bearing(to location: CLLocation) -> Uncertain<Double> {
            return Self.bearing(from: self, to: Uncertain<CLLocation>.from(location))
        }
    }

    extension Uncertain where T == CLLocationCoordinate2D {
        /// Creates an uncertain coordinate from a CLLocationCoordinate2D with accuracy.
        ///
        /// - Parameters:
        ///   - coordinate: The GPS coordinate.
        ///   - accuracy: The GPS accuracy in meters (used as scale parameter for Rayleigh distribution).
        /// - Returns: A new uncertain coordinate value.
        public static func coordinate(
            _ coordinate: CLLocationCoordinate2D,
            accuracy: CLLocationAccuracy
        ) -> Uncertain<CLLocationCoordinate2D> {
            return Uncertain<CLLocationCoordinate2D> {
                let earthRadius = 6_371_000.0  // meters
                let lat1 = coordinate.latitude * .pi / 180

                // Use Rayleigh distribution for radial distance from true location
                let radialDistance = Uncertain<Double>.rayleigh(scale: accuracy).sample()

                // Generate random direction (angle) uniformly
                let angle = Double.random(in: 0...(2 * .pi))

                // Convert polar coordinates to Cartesian offsets
                let northOffset = radialDistance * cos(angle)
                let eastOffset = radialDistance * sin(angle)

                // Convert meter offsets to lat/lon
                let latOffset = northOffset / earthRadius * 180 / .pi
                let lonOffset = eastOffset / (earthRadius * cos(lat1)) * 180 / .pi

                return CLLocationCoordinate2D(
                    latitude: coordinate.latitude + latOffset,
                    longitude: coordinate.longitude + lonOffset
                )
            }
        }

        /// Calculates the distance between two uncertain coordinates.
        ///
        /// - Parameters:
        ///   - from: The starting coordinate.
        ///   - to: The ending coordinate.
        /// - Returns: An uncertain distance value in meters.
        public static func distance(
            from: Uncertain<CLLocationCoordinate2D>,
            to: Uncertain<CLLocationCoordinate2D>
        ) -> Uncertain<CLLocationDistance> {
            return Uncertain<CLLocationDistance> {
                let coord1 = from.sample()
                let coord2 = to.sample()
                let loc1 = CLLocation(latitude: coord1.latitude, longitude: coord1.longitude)
                let loc2 = CLLocation(latitude: coord2.latitude, longitude: coord2.longitude)
                return loc1.distance(from: loc2)
            }
        }

        /// Calculates the bearing between two uncertain coordinates.
        ///
        /// - Parameters:
        ///   - from: The starting coordinate.
        ///   - to: The ending coordinate.
        /// - Returns: An uncertain bearing value in degrees (0-360).
        public static func bearing(
            from: Uncertain<CLLocationCoordinate2D>,
            to: Uncertain<CLLocationCoordinate2D>
        ) -> Uncertain<Double> {
            return Uncertain<Double> {
                let coord1 = from.sample()
                let coord2 = to.sample()

                let lat1 = coord1.latitude * .pi / 180
                let lat2 = coord2.latitude * .pi / 180
                let deltaLon = (coord2.longitude - coord1.longitude) * .pi / 180

                let x = sin(deltaLon) * cos(lat2)
                let y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(deltaLon)
                let bearing = atan2(x, y)

                // Convert to degrees and normalize to 0-360
                var degrees = bearing * 180 / .pi
                if degrees < 0 {
                    degrees += 360
                }
                return degrees
            }
        }

        /// Calculates the distance from this coordinate to another coordinate.
        ///
        /// - Parameter coordinate: The destination coordinate.
        /// - Returns: An uncertain distance value in meters.
        public func distance(to coordinate: Uncertain<CLLocationCoordinate2D>) -> Uncertain<
            CLLocationDistance
        > {
            return Self.distance(from: self, to: coordinate)
        }

        /// Calculates the distance from this coordinate to a specific coordinate.
        ///
        /// - Parameter coordinate: The destination coordinate.
        /// - Returns: An uncertain distance value in meters.
        public func distance(to coordinate: CLLocationCoordinate2D) -> Uncertain<CLLocationDistance>
        {
            return Uncertain<CLLocationDistance> {
                let coord1 = self.sample()
                let loc1 = CLLocation(latitude: coord1.latitude, longitude: coord1.longitude)
                let loc2 = CLLocation(
                    latitude: coordinate.latitude, longitude: coordinate.longitude)
                return loc1.distance(from: loc2)
            }
        }

        /// Calculates the bearing from this coordinate to another coordinate.
        ///
        /// - Parameter coordinate: The destination coordinate.
        /// - Returns: An uncertain bearing value in degrees (0-360).
        public func bearing(to coordinate: Uncertain<CLLocationCoordinate2D>) -> Uncertain<Double> {
            return Self.bearing(from: self, to: coordinate)
        }

        /// Calculates the bearing from this coordinate to a specific coordinate.
        ///
        /// - Parameter coordinate: The destination coordinate.
        /// - Returns: An uncertain bearing value in degrees (0-360).
        public func bearing(to coordinate: CLLocationCoordinate2D) -> Uncertain<Double> {
            return Uncertain<Double> {
                let coord1 = self.sample()

                let lat1 = coord1.latitude * .pi / 180
                let lat2 = coordinate.latitude * .pi / 180
                let deltaLon = (coordinate.longitude - coord1.longitude) * .pi / 180

                let x = sin(deltaLon) * cos(lat2)
                let y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(deltaLon)
                let bearing = atan2(x, y)

                // Convert to degrees and normalize to 0-360
                var degrees = bearing * 180 / .pi
                if degrees < 0 {
                    degrees += 360
                }
                return degrees
            }
        }
    }

    @available(iOS 14.0, macOS 11.0, watchOS 7.0, tvOS 14.0, *)
    extension Uncertain where T == CLLocationSpeed {
        /// Creates an uncertain speed from a CLLocation with speed uncertainty.
        ///
        /// - Parameters:
        ///   - location: The CLLocation with speed and course information.
        ///   - speedUncertaintyFactor: Factor to estimate speed uncertainty from horizontal accuracy (default: 0.1).
        ///   - minimumSpeedUncertainty: Minimum speed uncertainty in m/s (default: 0.1).
        /// - Returns: A new uncertain speed value in m/s.
        public static func speed(
            from location: CLLocation,
            speedUncertaintyFactor: Double = 0.1,
            minimumSpeedUncertainty: Double = 0.1
        ) -> Uncertain<CLLocationSpeed> {
            return Uncertain<CLLocationSpeed> {
                let baseSpeed = location.speed

                // If speed is negative, it's invalid
                guard baseSpeed >= 0 else {
                    return -1
                }

                // Model speed uncertainty (Core Location doesn't provide direct speed accuracy)
                // Use configurable estimate based on location accuracy
                let speedUncertainty = Swift.max(
                    minimumSpeedUncertainty, location.horizontalAccuracy * speedUncertaintyFactor)
                let uncertainSpeed = Swift.max(
                    0,
                    Uncertain<Double>.normal(mean: baseSpeed, standardDeviation: speedUncertainty)
                        .sample()
                )

                return uncertainSpeed
            }
        }

        /// Checks if this speed exceeds a given speed limit with evidence-based evaluation.
        ///
        /// - Parameter speedLimit: The speed limit in m/s.
        /// - Returns: Uncertain boolean evidence that this speed exceeds the limit.
        public func exceeds(_ speedLimit: CLLocationSpeed) -> Uncertain<Bool> {
            return self > speedLimit
        }

        /// Checks if this speed is within a given range.
        ///
        /// - Parameters:
        ///   - min: The minimum speed in m/s.
        ///   - max: The maximum speed in m/s.
        /// - Returns: Uncertain boolean evidence that this speed is within the range.
        public func isWithin(min: CLLocationSpeed, max: CLLocationSpeed) -> Uncertain<Bool> {
            return (self >= min) && (self <= max)
        }
    }

    @available(iOS 14.0, macOS 11.0, watchOS 7.0, tvOS 14.0, *)
    extension Uncertain where T == CLLocationDirection {
        /// Creates an uncertain course/heading from a CLLocation.
        ///
        /// - Parameters:
        ///   - location: The CLLocation with course information.
        ///   - courseUncertainty: The uncertainty in course/heading in degrees (default: 5.0).
        /// - Returns: A new uncertain course value in degrees.
        public static func course(
            from location: CLLocation,
            courseUncertainty: Double = 5.0
        ) -> Uncertain<CLLocationDirection> {
            return Uncertain<CLLocationDirection> {
                let baseCourse = location.course

                // If course is negative, it's invalid
                guard baseCourse >= 0 else {
                    return -1
                }

                // Model course uncertainty with configurable parameter
                let uncertainCourse = Uncertain<Double>.normal(
                    mean: baseCourse, standardDeviation: courseUncertainty
                ).sample()

                // Normalize to 0-360 degrees
                var normalizedCourse = uncertainCourse.truncatingRemainder(dividingBy: 360)
                if normalizedCourse < 0 {
                    normalizedCourse += 360
                }
                return normalizedCourse
            }
        }

        /// Calculates the angular difference between this direction and another direction.
        ///
        /// Returns the shortest angular distance between two directions,
        /// accounting for the circular nature of compass directions.
        ///
        /// - Parameter direction: The other direction in degrees.
        /// - Returns: An uncertain angular difference in degrees (-180 to 180).
        public func angularDifference(to direction: CLLocationDirection) -> Uncertain<Double> {
            return self.map { thisCourse in
                var diff = direction - thisCourse

                // Normalize to [-180, 180] range
                while diff > 180 {
                    diff -= 360
                }
                while diff < -180 {
                    diff += 360
                }

                return diff
            }
        }

        /// Calculates the angular difference between this direction and another uncertain direction.
        ///
        /// - Parameter direction: The other uncertain direction.
        /// - Returns: An uncertain angular difference in degrees (-180 to 180).
        public func angularDifference(to direction: Uncertain<CLLocationDirection>) -> Uncertain<
            Double
        > {
            return Uncertain<Double> {
                let thisCourse = self.sample()
                let otherCourse = direction.sample()

                var diff = otherCourse - thisCourse

                // Normalize to [-180, 180] range
                while diff > 180 {
                    diff -= 360
                }
                while diff < -180 {
                    diff += 360
                }

                return diff
            }
        }

        /// Checks if this direction is within a given angular range of another direction.
        ///
        /// - Parameters:
        ///   - direction: The target direction in degrees.
        ///   - tolerance: The angular tolerance in degrees.
        /// - Returns: Uncertain boolean evidence that this direction is within the tolerance.
        public func isWithin(_ tolerance: Double, of direction: CLLocationDirection) -> Uncertain<
            Bool
        > {
            let diff = self.angularDifference(to: direction)
            return diff.map { abs($0) <= tolerance }
        }
    }
#endif

// MARK: -

/// Context for memoizing samples within a single evaluation to ensure
/// shared variables produce the same sample value
internal class SampleContext {
    private var memoizedValues: [UUID: Any] = [:]

    func getValue<T>(for id: UUID) -> T? {
        return memoizedValues[id] as? T
    }

    func setValue<T>(_ value: T, for id: UUID) {
        memoizedValues[id] = value
    }
}

/// Computation graph node for lazy evaluation using indirect enum
internal indirect enum ComputationNode<T>: Sendable where T: Sendable {
    /// Leaf node representing a direct sampling function with unique ID
    case leaf(id: UUID, sample: @Sendable () -> T)

    /// Binary operation node for combining two uncertain values
    case binaryOp(
        left: ComputationNode<T>,
        right: ComputationNode<T>,
        operation: @Sendable (T, T) -> T
    )

    /// Evaluates the computation graph node with memoization context
    func evaluate(in context: inout SampleContext) -> T {
        switch self {
        case .leaf(let id, let sample):
            if let cached: T = context.getValue(for: id) {
                return cached
            }
            let value = sample()
            context.setValue(value, for: id)
            return value
        case .binaryOp(let left, let right, let operation):
            return operation(left.evaluate(in: &context), right.evaluate(in: &context))
        }
    }

    /// Evaluates the computation graph node in a new context
    func evaluate() -> T {
        var context = SampleContext()
        return evaluate(in: &context)
    }
}

/// Specialized computation node for comparisons that return Bool
internal indirect enum ComparisonNode<T>: Sendable where T: Sendable & Comparable {
    /// Comparison operation node returning uncertain boolean evidence
    case comparison(
        left: ComputationNode<T>,
        threshold: T,
        comparison: @Sendable (T, T) -> Bool
    )

    /// Evaluates the comparison node with memoization context
    func evaluate(in context: inout SampleContext) -> Bool {
        switch self {
        case .comparison(let left, let threshold, let comparison):
            return comparison(left.evaluate(in: &context), threshold)
        }
    }

    /// Evaluates the comparison node in a new context
    func evaluate() -> Bool {
        var context = SampleContext()
        return evaluate(in: &context)
    }
}

/// Specialized computation node for equality comparisons that return Bool
internal indirect enum EqualityNode<T>: Sendable where T: Sendable & Equatable {
    /// Equality operation node returning uncertain boolean evidence
    case comparison(
        left: ComputationNode<T>,
        threshold: T,
        comparison: @Sendable (T, T) -> Bool
    )

    /// Evaluates the equality node with memoization context
    func evaluate(in context: inout SampleContext) -> Bool {
        switch self {
        case .comparison(let left, let threshold, let comparison):
            return comparison(left.evaluate(in: &context), threshold)
        }
    }

    /// Evaluates the equality node in a new context
    func evaluate() -> Bool {
        var context = SampleContext()
        return evaluate(in: &context)
    }
}
