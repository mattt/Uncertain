import Foundation
import Testing

@testable import Uncertain

/// Checks that distributions made from generators with the same seed give the same samples,
/// and that distributions made from generators with different seeds do not.
func expectReproducible<V: Equatable & Sendable>(
    sampleCount: Int = 20,
    sourceLocation: SourceLocation = #_sourceLocation,
    _ makeDistribution: (SplitMix64) -> Uncertain<V>
) {
    let first = Array(makeDistribution(SplitMix64(seed: 1)).prefix(sampleCount))
    let repeated = Array(makeDistribution(SplitMix64(seed: 1)).prefix(sampleCount))
    let other = Array(makeDistribution(SplitMix64(seed: 2)).prefix(sampleCount))
    #expect(first == repeated, sourceLocation: sourceLocation)
    #expect(first != other, sourceLocation: sourceLocation)
}

/// A value-type generator, to check that each distribution keeps its own copy.
struct LinearCongruentialGenerator: RandomNumberGenerator, Sendable {
    var state: UInt64

    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}

@Suite("Seeded distributions")
struct SeededDistributionTests {
    @Test("SplitMix64 matches the reference sequence")
    func testSplitMix64ReferenceSequence() {
        // The first outputs of the reference implementation for seed 0
        let generator = SplitMix64(seed: 0)
        #expect(generator.next() == 0xE220_A839_7B1D_CDAF)
        #expect(generator.next() == 0x6E78_9E6A_A1B9_65F4)
        #expect(generator.next() == 0x06C4_5D18_8009_454F)
    }

    @Test("Continuous distributions are reproducible")
    func testContinuousDistributions() {
        expectReproducible { .normal(mean: 0, standardDeviation: 1, using: $0) }
        expectReproducible { .uniform(min: 0, max: 1, using: $0) }
        expectReproducible { .exponential(rate: 2, using: $0) }
        expectReproducible { .kumaraswamy(a: 2, b: 3, using: $0) }
        expectReproducible { .rayleigh(scale: 1, using: $0) }
    }

    @Test("Discrete distributions are reproducible")
    func testDiscreteDistributions() {
        expectReproducible { Uncertain<Double>.bernoulli(probability: 0.5, using: $0) }
        expectReproducible { Uncertain<Int>.binomial(trials: 10, probability: 0.5, using: $0) }
        // Lambda below 10 uses multiplication, and 10 or more uses transformed rejection
        expectReproducible { Uncertain<Int>.poisson(lambda: 3.5, using: $0) }
        expectReproducible { Uncertain<Int>.poisson(lambda: 10, using: $0) }
        expectReproducible { Uncertain<Int>.poisson(lambda: 1000, using: $0) }
        expectReproducible { Uncertain<String>.categorical(["a": 0.2, "b": 0.3, "c": 0.5], using: $0)! }
        expectReproducible { Uncertain<Int>.empirical([1, 2, 3, 4, 5, 6], using: $0)! }
    }

    @Test("Mixture distributions are reproducible")
    func testMixtureDistribution() {
        expectReproducible { generator in
            Uncertain<Double>.mixture(
                of: [
                    .normal(mean: 0, standardDeviation: 1, using: generator),
                    .uniform(min: 5, max: 6, using: generator),
                ],
                weights: [0.3, 0.7],
                using: generator
            )
        }
    }

    @Test("Distributions that share a generator draw from one stream")
    func testSharedReferenceGenerator() {
        let generator = SplitMix64(seed: 1)
        let x = Uncertain<Double>.normal(mean: 0, standardDeviation: 1, using: generator)
        let y = Uncertain<Double>.normal(mean: 0, standardDeviation: 1, using: generator)

        let xSamples = Array(x.prefix(10))
        let ySamples = Array(y.prefix(10))
        #expect(xSamples != ySamples)

        // Together, the two distributions read the stream that one distribution reads alone
        let alone = Array(
            Uncertain<Double>.normal(mean: 0, standardDeviation: 1, using: SplitMix64(seed: 1)).prefix(20)
        )
        #expect(xSamples + ySamples == alone)
    }

    @Test("Each distribution keeps its own copy of a value-type generator")
    func testCopiedValueGenerator() {
        let generator = LinearCongruentialGenerator(state: 1)
        let x = Uncertain<Double>.uniform(min: 0, max: 1, using: generator)
        let y = Uncertain<Double>.uniform(min: 0, max: 1, using: generator)

        #expect(Array(x.prefix(10)) == Array(y.prefix(10)))
    }

    @Test("Concurrent samples give the same values as sequential samples")
    func testConcurrentSampling() async {
        let sampleCount = 1_000
        let sequential = Array(
            Uncertain<Double>.normal(mean: 0, standardDeviation: 1, using: SplitMix64(seed: 3)).prefix(sampleCount)
        )

        let distribution = Uncertain<Double>.normal(mean: 0, standardDeviation: 1, using: SplitMix64(seed: 3))
        let concurrent = await withTaskGroup(of: Double.self) { group in
            for _ in 0 ..< sampleCount {
                group.addTask { distribution.sample() }
            }
            return await group.reduce(into: [Double]()) { $0.append($1) }
        }

        #expect(concurrent.sorted() == sequential.sorted())
    }
}

#if canImport(CoreLocation)
    import CoreLocation

    @Suite("Seeded Core Location distributions")
    struct SeededCoreLocationTests {
        let location = CLLocation(
            coordinate: CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
            altitude: 10.0,
            horizontalAccuracy: 5.0,
            verticalAccuracy: 2.0,
            course: 90.0,
            speed: 15.0,
            timestamp: Date()
        )

        @Test("Location distributions are reproducible")
        func testLocationDistributions() {
            let location = self.location
            @Sendable func components(_ location: CLLocation) -> [Double] {
                [location.coordinate.latitude, location.coordinate.longitude, location.altitude]
            }

            expectReproducible { Uncertain<CLLocation>.from(location, using: $0).map(components) }
            expectReproducible { generator in
                Uncertain<CLLocation>.from(
                    location,
                    horizontalAccuracy: 20.0,
                    verticalAccuracy: 4.0,
                    using: generator
                ).map(components)
            }
            expectReproducible { generator in
                Uncertain<CLLocation>.location(
                    coordinate: location.coordinate,
                    horizontalAccuracy: 10.0,
                    verticalAccuracy: 3.0,
                    altitude: 50.0,
                    using: generator
                ).map(components)
            }
            expectReproducible { generator in
                Uncertain<CLLocationCoordinate2D>.coordinate(location.coordinate, accuracy: 8.0, using: generator)
                    .map { [$0.latitude, $0.longitude] }
            }
        }

        @Test("Speed and course distributions are reproducible")
        func testSpeedAndCourseDistributions() {
            guard #available(iOS 14.0, macOS 11.0, watchOS 7.0, tvOS 14.0, *) else { return }
            let location = CLLocation(
                coordinate: self.location.coordinate,
                altitude: 0,
                horizontalAccuracy: 10.0,
                verticalAccuracy: 5.0,
                course: 90.0,
                courseAccuracy: 5.0,
                speed: 15.0,
                speedAccuracy: 1.0,
                timestamp: Date()
            )

            expectReproducible { Uncertain<CLLocationSpeed>.speed(from: location, using: $0) }
            expectReproducible { Uncertain<CLLocationDirection>.course(from: location, using: $0) }
            expectReproducible {
                Uncertain<CLLocationDirection>.course(from: location, courseAccuracy: 10.0, using: $0)
            }
        }
    }
#endif
