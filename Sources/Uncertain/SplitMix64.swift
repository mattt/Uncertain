import Foundation

/// A seedable random number generator that uses the SplitMix64 algorithm.
///
/// Pass a `SplitMix64` instance to a distribution constructor's
/// `using:` parameter to make its samples reproducible:
///
/// ```swift
/// let generator = SplitMix64(seed: 42)
/// let x = Uncertain<Double>.normal(mean: 0, standardDeviation: 1, using: generator)
/// let y = Uncertain<Double>.uniform(min: 0, max: 1, using: generator)
/// ```
///
/// `SplitMix64` is a class, so distributions made from one instance
/// draw from one shared stream, in the order that they sample.
/// Sequential sampling gives the same values on every run.
/// The generator is safe to use from several threads at the same time,
/// but then the order of the draws, and so the values, can change between runs.
///
/// For a given seed, the output sequence is the one that
/// Sebastiano Vigna's reference implementation of SplitMix64 produces,
/// and it does not change between releases of this package.
///
/// - Important: `SplitMix64` is not cryptographically secure.
///   Use `SystemRandomNumberGenerator` when values must be unpredictable.
public final class SplitMix64: RandomNumberGenerator, @unchecked Sendable {
    private var state: UInt64
    private let lock = NSLock()

    /// Creates a generator with the given seed.
    ///
    /// - Parameter seed: The initial state of the generator.
    ///   Generators with the same seed produce the same sequence.
    public init(seed: UInt64) {
        state = seed
    }

    /// Returns the next value in the generator's sequence.
    public func next() -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
