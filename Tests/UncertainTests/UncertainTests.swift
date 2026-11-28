import Foundation
import Testing

@testable import Uncertain

@Suite("Paper-Faithful Uncertain<T> Tests")
struct UncertainTests {

    // MARK: - Core Concepts

    @Test("Evidence-based conditionals prevent uncertainty bugs")
    func testEvidenceBasedConditionals() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Temperature sensor with ±2°C accuracy
        let temperature = Uncertain<Double>.normal(mean: 0.5, standardDeviation: 2.0, using: rng)
        let freezingPoint = 0.0

        // Test that we get an Uncertain<Bool> back
        let freezingEvidence = temperature < freezingPoint
        #expect(type(of: freezingEvidence) == Uncertain<Bool>.self)

        // Test different confidence levels
        let highConfidence = freezingEvidence.probability(exceeds: 0.95)
        let mediumConfidence = freezingEvidence.probability(exceeds: 0.5)
        let lowConfidence = freezingEvidence.probability(exceeds: 0.1)

        // These should be boolean decisions
        #expect(type(of: highConfidence) == Bool.self)
        #expect(type(of: mediumConfidence) == Bool.self)
        #expect(type(of: lowConfidence) == Bool.self)

        // With temperature mean=0.5°C and std=2°C, P(freezing) ≈ 0.40, so freezing should be:
        // - Very unlikely at 95% confidence (false)
        // - Likely at low confidence (true)
        #expect(highConfidence == false)
        #expect(lowConfidence == true)
    }

    @Test("Implicit conditionals using ~ operator")
    func testImplicitConditionals() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Speed measurement with uncertainty
        let speed = Uncertain<Double>.normal(mean: 65.0, standardDeviation: 3.0, using: rng)
        let speedLimit = 60.0

        let speedingEvidence = speed > speedLimit

        // Test implicit conditional (~ operator)
        let implicitResult = ~speedingEvidence
        let explicitResult = speedingEvidence.probability(exceeds: 0.5)

        // They should be equivalent
        #expect(implicitResult == explicitResult)

        // With mean=65 and std=3, should be speeding more likely than not
        #expect(implicitResult == true)
    }

    @Test("GPS Walking Speed Example from Paper")
    func testGPSWalkingSpeedExample() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // GPS readings with 4-meter accuracy (from the paper)
        let location1 = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 4.0, using: rng)
        let location2 = Uncertain<Double>.normal(mean: 100.0, standardDeviation: 4.0, using: rng)

        // Calculate walking speed with uncertainty propagation
        let distance = location2 - location1
        let time = 30.0  // 30 seconds
        let speed = distance / time
        let speedMph = speed * 2.237  // Convert to mph

        // Evidence-based decisions
        let fastEvidence = speedMph > 4.0

        // Test that uncertainty propagates correctly
        let samples = speedMph.prefix(1000)
        let count = Array(samples).count
        let mean = samples.reduce(0, +) / Double(count)

        // Expected speed: 100m in 30s = 3.33 m/s = 7.45 mph
        #expect(abs(mean - 7.45) < 1.0)

        // Test evidence-based conditional
        let confident = fastEvidence.probability(exceeds: 0.8)
        #expect(confident == true)  // Should be confident about walking fast
    }

    @Test("Speeding Ticket Example from Paper")
    func testSpeedingTicketExample() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // True speed is 56 mph, but measurement has error
        let trueSpeed = 56.0
        let measurementError = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 2.0, using: rng)
        let measuredSpeed = Uncertain<Double> { trueSpeed + measurementError.sample() }

        let speedLimit = 60.0
        let speedingEvidence = measuredSpeed > speedLimit

        // Different enforcement strategies
        let lenientOfficer = speedingEvidence.probability(exceeds: 0.95)
        let standardOfficer = ~speedingEvidence
        let strictOfficer = speedingEvidence.probability(exceeds: 0.2)

        // With true speed 56 mph and 2 mph error, speed limit 60 mph, P(speeding) ≈ 0.023:
        // - Lenient (95% confident): Should not ticket
        // - Standard (50% confident): Should not ticket
        // - Strict (20% confident): Should not ticket
        #expect(lenientOfficer == false)
        #expect(standardOfficer == false)
        #expect(strictOfficer == false)
    }

    @Test("Comparison operators return Uncertain<Bool>")
    func testComparisonOperatorsReturnUncertainBool() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        let value = Uncertain<Double>.normal(mean: 10.0, standardDeviation: 2.0, using: rng)
        let threshold = 12.0

        let gtEvidence = value > threshold
        let ltEvidence = value < threshold
        let geEvidence = value >= threshold
        let leEvidence = value <= threshold

        // All should return Uncertain<Bool>
        #expect(type(of: gtEvidence) == Uncertain<Bool>.self)
        #expect(type(of: ltEvidence) == Uncertain<Bool>.self)
        #expect(type(of: geEvidence) == Uncertain<Bool>.self)
        #expect(type(of: leEvidence) == Uncertain<Bool>.self)

        // Test that they can be used in evidence-based conditionals
        let gtResult = gtEvidence.probability(exceeds: 0.5)
        let ltResult = ltEvidence.probability(exceeds: 0.5)

        // With mean=10, std=2, threshold=12:
        // - P(value > 12) should be low
        // - P(value < 12) should be high
        #expect(gtResult == false)
        #expect(ltResult == true)
    }

    // MARK: - Computation Graph Tests

    @Test("Computation graph builds correctly")
    func testComputationGraphBuilding() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        let x = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0, using: rng)
        let y = Uncertain<Double>.normal(mean: 3.0, standardDeviation: 1.0, using: rng)

        // Build computation graph
        let sum = x + y
        let product = x * y
        let complex = (x + y) * 2.0 - product

        // Test that operations work correctly
        let sumMean = sum.expectedValue(sampleCount: 1000)
        let productMean = product.expectedValue(sampleCount: 1000)
        let complexMean = complex.expectedValue(sampleCount: 1000)

        #expect(abs(sumMean - 8.0) < 0.5)  // 5 + 3
        #expect(abs(productMean - 15.0) < 1.0)  // 5 * 3
        #expect(abs(complexMean - 1.0) < 1.0)  // (5+3)*2 - 5*3 = 16 - 15 = 1
    }

    @Test("Lazy evaluation with computation graph")
    func testLazyEvaluation() {
        // Use a simpler approach to test lazy evaluation
        let x = Uncertain<Double> { 10.0 }
        let y = Uncertain<Double> { 5.0 }

        // Build computation graph
        let sum = x + y
        let evidence = sum > 12.0

        // Test that the computation graph works correctly
        let result = evidence.probability(exceeds: 0.5)

        #expect(result == true)  // 10 + 5 = 15 > 12

        // Test more complex graph
        let complex = (x + y) * 2.0 - x
        let complexEvidence = complex > 25.0
        let complexResult = complexEvidence.probability(exceeds: 0.5)

        // (10 + 5) * 2 - 10 = 30 - 10 = 20 < 25
        #expect(complexResult == false)
    }

    // MARK: - Hypothesis Testing Tests

    @Test("Hypothesis testing with different confidence levels")
    func testHypothesisTestingConfidenceLevels() {
        // Create a distribution where we know the true probability.
        // A fixed seed keeps the decisions the same on every run.
        let rng = SplitMix64(seed: 1)
        let biasedCoin = Uncertain<Double>.bernoulli(probability: 0.7, using: rng)

        // Test different confidence levels
        let highConfidence = biasedCoin.probability(exceeds: 0.6, confidenceLevel: 0.95)
        let mediumConfidence = biasedCoin.probability(exceeds: 0.6, confidenceLevel: 0.90)
        let lowConfidence = biasedCoin.probability(exceeds: 0.6, confidenceLevel: 0.80)

        // With true probability 0.7, testing exceeds 0.6 should be true
        #expect(highConfidence == true)
        #expect(mediumConfidence == true)
        #expect(lowConfidence == true)

        // But testing exceeds 0.8 should be false
        let impossibleThreshold = biasedCoin.probability(exceeds: 0.8, confidenceLevel: 0.95)
        #expect(impossibleThreshold == false)
    }

    @Test("SPRT-inspired hypothesis testing efficiency")
    func testHypothesisTestingEfficiency() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Create clear evidence case
        let certainTrue = Uncertain<Bool> { true }
        let certainFalse = Uncertain<Bool> { false }

        // These should be decided quickly with few samples
        let trueResult = certainTrue.probability(exceeds: 0.5)
        let falseResult = certainFalse.probability(exceeds: 0.5)

        #expect(trueResult == true)
        #expect(falseResult == false)

        // Test edge case near threshold
        let nearThreshold = Uncertain<Double>.bernoulli(probability: 0.51, using: rng)
        let edgeResult = nearThreshold.probability(exceeds: 0.5)

        // Should still make a decision (though might need more samples)
        #expect(type(of: edgeResult) == Bool.self)
    }

    // MARK: - Distribution Tests (Paper-Faithful)

    @Test("Normal distribution for uncertainty modeling")
    func testNormalDistributionForUncertainty() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // GPS accuracy example
        let gpsReading = Uncertain<Double>.normal(mean: 100.0, standardDeviation: 5.0, using: rng)

        // Test uncertainty propagation
        // 10,000 samples put the tolerance on the mean at 10 standard errors
        let samples = gpsReading.prefix(10_000)
        let count = Array(samples).count
        let mean = samples.reduce(0, +) / Double(count)
        let variance = samples.reduce(0) { acc, x in acc + (x - mean) * (x - mean) } / Double(count)
        let stdDev = sqrt(variance)

        #expect(abs(mean - 100.0) < 0.5)
        #expect(abs(stdDev - 5.0) < 0.5)
    }

    @Test("Uniform distribution for modeling ranges")
    func testUniformDistributionForRanges() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        let sensorRange = Uncertain<Double>.uniform(min: 0.0, max: 100.0, using: rng)
        let samples = sensorRange.prefix(1000)

        // All samples should be in range
        #expect(samples.allSatisfy { $0 >= 0.0 && $0 <= 100.0 })

        // Mean should be approximately 50
        let count = Array(samples).count
        let mean = samples.reduce(0, +) / Double(count)
        #expect(abs(mean - 50.0) < 5.0)
    }

    @Test("Normal distribution reaches its tails")
    func testNormalDistributionTails() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        let standardNormal = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 1.0, using: rng)

        // P(|Z| > 4) ≈ 6.334e-5, so 500,000 samples contain about 32 such values.
        // Under a Poisson approximation, P(count == 0) ≈ e^-31.7 ≈ 2e-14 and P(count >= 150) < 1e-40.
        let sampleCount = 500_000
        var beyondFour = 0
        for _ in 0 ..< sampleCount where abs(standardNormal.sample()) > 4.0 {
            beyondFour += 1
        }

        #expect(beyondFour > 0)
        #expect(beyondFour < 150)
    }

    @Test("Exponential distribution produces finite positive values")
    func testExponentialDistribution() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        let rate = 2.0
        let waitingTime = Uncertain<Double>.exponential(rate: rate, using: rng)

        let samples = (0 ..< 100_000).map { _ in waitingTime.sample() }
        #expect(samples.allSatisfy { $0.isFinite && $0 > 0 })

        // The standard error of the mean is 0.5 / sqrt(100,000) ≈ 0.0016,
        // so the tolerance of 0.016 is 10 standard errors (P(failure) ≈ 2e-23).
        let mean = samples.reduce(0, +) / Double(samples.count)
        #expect(abs(mean - 1.0 / rate) < 0.016)
    }

    @Test("Certain and impossible outcomes never change")
    func testCertainAndImpossibleOutcomes() {
        let sampleCount = 10_000

        let alwaysTrue = Uncertain<Double>.bernoulli(probability: 1.0)
        let neverTrue = Uncertain<Double>.bernoulli(probability: 0.0)
        #expect((0 ..< sampleCount).allSatisfy { _ in alwaysTrue.sample() })
        #expect((0 ..< sampleCount).allSatisfy { _ in !neverTrue.sample() })

        let allSuccesses = Uncertain<Int>.binomial(trials: 10, probability: 1.0)
        let noSuccesses = Uncertain<Int>.binomial(trials: 10, probability: 0.0)
        #expect((0 ..< sampleCount).allSatisfy { _ in allSuccesses.sample() == 10 })
        #expect((0 ..< sampleCount).allSatisfy { _ in noSuccesses.sample() == 0 })

        let category = Uncertain<String>.categorical(["never": 0.0, "sometimes": 0.4, "often": 0.6])!
        #expect((0 ..< sampleCount).allSatisfy { _ in category.sample() != "never" })

        let mixture = Uncertain<Double>.mixture(
            of: [.point(0.0), .point(1.0), .point(2.0)],
            weights: [0.0, 1.0, 0.0]
        )
        #expect((0 ..< sampleCount).allSatisfy { _ in mixture.sample() == 1.0 })
    }

    @Test("Poisson distribution with zero lambda is always zero")
    func testPoissonDistributionWithZeroLambda() {
        let poisson = Uncertain<Int>.poisson(lambda: 0)
        #expect(poisson.prefix(1000).allSatisfy { $0 == 0 })
    }

    @Test("Poisson distribution has the correct mean and variance for large lambda", arguments: [1000.0, 10_000.0])
    func testPoissonDistributionMeanAndVarianceForLargeLambda(lambda: Double) {
        let n = 10_000
        let samples = Array(Uncertain<Int>.poisson(lambda: lambda).prefix(n)).map(Double.init)
        let mean = samples.reduce(0, +) / Double(n)
        let variance = samples.reduce(0) { acc, x in acc + (x - mean) * (x - mean) } / Double(n - 1)

        // The sum of the samples is Poisson with mean n * lambda,
        // so a margin of 7 standard errors on the mean fails with probability 2.6e-12.
        #expect(abs(mean - lambda) < 7 * (lambda / Double(n)).squareRoot())
        // (n - 1) * variance / lambda is close to chi-squared with n - 1 degrees of freedom,
        // so a margin of 10% on the variance fails with probability 3.8e-12.
        #expect(abs(variance / lambda - 1) < 0.1)
    }

    @Test("Poisson distribution matches the probability mass function for moderate lambda")
    func testPoissonDistributionGoodnessOfFit() {
        let lambda = 30.0
        let n = 100_000
        // Bins are k <= 14, each k from 15 through 49, and k >= 50.
        let low = 14
        let high = 50
        var observed = [Int](repeating: 0, count: high - low + 1)
        for k in Uncertain<Int>.poisson(lambda: lambda).prefix(n) {
            observed[min(max(k, low), high) - low] += 1
        }

        func pmf(_ k: Int) -> Double {
            exp(Double(k) * log(lambda) - lambda - lgamma(Double(k) + 1))
        }
        var probabilities = (low ... high).map(pmf)
        probabilities[0] = (0 ... low).map(pmf).reduce(0, +)
        probabilities[probabilities.count - 1] = 1 - (0 ..< high).map(pmf).reduce(0, +)

        let statistic = zip(observed, probabilities).reduce(0.0) { acc, pair in
            let expected = Double(n) * pair.1
            return acc + (Double(pair.0) - expected) * (Double(pair.0) - expected) / expected
        }
        // The statistic is close to chi-squared with 36 degrees of freedom,
        // which is above 130 with probability 1.5e-12.
        #expect(statistic < 130)
    }

    // MARK: - Arithmetic Operations with Uncertainty Propagation

    @Test("Uncertainty propagation through arithmetic")
    func testUncertaintyPropagationThroughArithmetic() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        let measurement1 = Uncertain<Double>.normal(mean: 10.0, standardDeviation: 1.0, using: rng)
        let measurement2 = Uncertain<Double>.normal(mean: 20.0, standardDeviation: 2.0, using: rng)

        let sum = measurement1 + measurement2
        let difference = measurement2 - measurement1
        let product = measurement1 * measurement2
        let quotient = measurement2 / measurement1

        // Test means
        let sumMean = sum.expectedValue(sampleCount: 1000)
        let diffMean = difference.expectedValue(sampleCount: 1000)
        let productMean = product.expectedValue(sampleCount: 1000)
        let quotientMean = quotient.expectedValue(sampleCount: 1000)

        #expect(abs(sumMean - 30.0) < 0.5)  // 10 + 20
        #expect(abs(diffMean - 10.0) < 0.5)  // 20 - 10
        #expect(abs(productMean - 200.0) < 10.0)  // 10 * 20
        #expect(abs(quotientMean - 2.0) < 0.2)  // 20 / 10

        // Test that uncertainty increases appropriately
        let sumStd = sum.standardDeviation(sampleCount: 1000)
        let diffStd = difference.standardDeviation(sampleCount: 1000)

        // For independent normal variables:
        // Var(X + Y) = Var(X) + Var(Y) -> Std(X + Y) = sqrt(1² + 2²) = sqrt(5) ≈ 2.24
        // Var(X - Y) = Var(X) + Var(Y) -> Std(X - Y) = sqrt(1² + 2²) = sqrt(5) ≈ 2.24
        #expect(abs(sumStd - 2.24) < 0.3)
        #expect(abs(diffStd - 2.24) < 0.3)
    }

    // MARK: - Error Cases and Edge Conditions

    @Test("Handle edge cases in evidence evaluation")
    func testEdgeCasesInEvidenceEvaluation() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Always true case
        let alwaysTrue = Uncertain<Bool> { true }
        #expect(alwaysTrue.probability(exceeds: 0.5) == true)
        #expect(alwaysTrue.probability(exceeds: 0.99) == true)
        #expect(~alwaysTrue == true)

        // Always false case
        let alwaysFalse = Uncertain<Bool> { false }
        #expect(alwaysFalse.probability(exceeds: 0.5) == false)
        #expect(alwaysFalse.probability(exceeds: 0.01) == false)
        #expect(~alwaysFalse == false)

        // Edge case: exactly at threshold
        let exactlyHalf = Uncertain<Double>.bernoulli(probability: 0.5, using: rng)
        let result = exactlyHalf.probability(exceeds: 0.5)
        // With true probability ~0.5, the result could be either true or false
        // depending on sampling variation, so we just check it's a boolean
        #expect(type(of: result) == Bool.self)
    }

    // MARK: - Performance and Efficiency Tests

    @Test("Efficient sampling with hypothesis testing")
    func testEfficientSamplingWithHypothesisTesting() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Test that clear cases are decided quickly
        let clearCase = Uncertain<Bool> { true }

        // This should work efficiently even with large max samples
        let result = clearCase.probability(exceeds: 0.5, maxSamples: 10000)
        #expect(result == true)

        // Test with uncertain case
        let uncertainCase = Uncertain<Double>.bernoulli(probability: 0.51, using: rng)
        let uncertainResult = uncertainCase.probability(exceeds: 0.5, maxSamples: 1000)

        // Should still produce a result (though might use more samples)
        #expect(type(of: uncertainResult) == Bool.self)
    }

    // MARK: - Paper Examples Integration Tests

    @Test("Complete GPS example from paper")
    func testCompleteGPSExample() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Simulate the complete GPS walking scenario from the paper
        let startLocation = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 4.0, using: rng)
        let endLocation = Uncertain<Double>.normal(mean: 100.0, standardDeviation: 4.0, using: rng)

        let distance = endLocation - startLocation
        let time = 30.0  // 30 seconds
        let speed = distance / time  // m/s
        let speedMph = speed * 2.237

        // Walking speed thresholds
        let slowWalking = 3.0  // mph
        let fastWalking = 4.0  // mph

        let slowEvidence = speedMph < slowWalking
        let fastEvidence = speedMph > fastWalking

        // Test evidence-based decisions
        let definitelyFast = fastEvidence.probability(exceeds: 0.9)
        let probablyFast = ~fastEvidence
        let definitelySlow = slowEvidence.probability(exceeds: 0.9)

        // With 100m in 30s ≈ 7.45 mph, should be fast
        #expect(definitelyFast == true)
        #expect(probablyFast == true)
        #expect(definitelySlow == false)
    }

    @Test("Uncertainty bug prevention")
    func testUncertaintyBugPrevention() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Demonstrate how evidence-based conditionals prevent bugs
        let sensorReading = Uncertain<Double>.normal(mean: 50.0, standardDeviation: 10.0, using: rng)
        let threshold = 55.0

        // OLD WAY (uncertainty bug): single sample
        let singleSample = sensorReading.sample()
        let buggyDecision = singleSample > threshold

        // NEW WAY (paper-faithful): evidence-based
        let evidence = sensorReading > threshold
        let safeDecision = evidence.probability(exceeds: 0.8)

        // The single sample could be anything, but evidence-based should be consistent
        #expect(type(of: buggyDecision) == Bool.self)  // Could be true or false randomly
        #expect(type(of: safeDecision) == Bool.self)

        // With mean=50, std=10, threshold=55, P(X > 55) ≈ 0.31
        // So probability(exceeds: 0.8) should be false
        #expect(safeDecision == false)
    }

    // MARK: - Paper-Faithful Improvements Tests

    @Test("Shared variable sampling with memoization")
    func testSharedVariableSampling() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // This test ensures that shared variables sample consistently within a single evaluation
        let x = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0, using: rng)

        // The paper states: "let b = a + a" should use the same sample of 'a' twice
        let doubleX = x + x
        let _ = x + x  // This should behave the same as doubleX

        // Test that x + x is equivalent to 2 * x for the same sample
        let timesTwo = x * 2.0

        // These should have the same distribution properties
        let doubleMean = doubleX.expectedValue(sampleCount: 1000)
        let timesTwoMean = timesTwo.expectedValue(sampleCount: 1000)

        #expect(abs(doubleMean - timesTwoMean) < 0.5)
        #expect(abs(doubleMean - 10.0) < 0.5)  // 2 * 5.0

        // Test more complex shared variable case
        let complex = (x + x) * (x - x)  // Should be (2x) * 0 = 0
        let complexMean = complex.expectedValue(sampleCount: 1000)

        // This should be very close to 0 if memoization works correctly
        #expect(abs(complexMean) < 0.5)
    }

    @Test("Point-mass constructor for constants")
    func testPointMassConstructor() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Test the paper's point-mass constructor
        let constantValue = Uncertain<Double>.point(42.0)
        let stringConstant = Uncertain<String>.point("hello")
        let boolConstant = Uncertain<Bool>.point(true)

        // These should always return the same value
        for _ in 0 ..< 10 {
            #expect(constantValue.sample() == 42.0)
            #expect(stringConstant.sample() == "hello")
            #expect(boolConstant.sample() == true)
        }

        // Test that point-mass values work in arithmetic
        let x = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0, using: rng)
        let constantTen = Uncertain<Double>.point(10.0)
        let sum = x + constantTen

        let sumMean = sum.expectedValue(sampleCount: 1000)
        #expect(abs(sumMean - 15.0) < 0.5)  // 5 + 10
    }

    @Test("Logical operators for Uncertain<Bool>")
    func testLogicalOperatorsForUncertainBool() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Create boolean distributions
        let mostlyTrue = Uncertain<Double>.bernoulli(probability: 0.8, using: rng)
        let mostlyFalse = Uncertain<Double>.bernoulli(probability: 0.2, using: rng)
        let alwaysTrue = Uncertain<Bool>.point(true)
        let alwaysFalse = Uncertain<Bool>.point(false)

        // Test AND operator
        let andResult1 = mostlyTrue && mostlyTrue
        let andResult2 = mostlyTrue && mostlyFalse
        let andResult3 = alwaysTrue && alwaysTrue
        let andResult4 = alwaysFalse && alwaysTrue

        #expect(type(of: andResult1) == Uncertain<Bool>.self)
        #expect(type(of: andResult2) == Uncertain<Bool>.self)
        #expect(andResult3.sample() == true)
        #expect(andResult4.sample() == false)

        // Test OR operator
        let orResult1 = mostlyTrue || mostlyFalse
        let orResult2 = mostlyFalse || mostlyFalse
        let orResult3 = alwaysTrue || alwaysFalse
        let orResult4 = alwaysFalse || alwaysFalse

        #expect(type(of: orResult1) == Uncertain<Bool>.self)
        #expect(type(of: orResult2) == Uncertain<Bool>.self)
        #expect(orResult3.sample() == true)
        #expect(orResult4.sample() == false)

        // Test NOT operator
        let notResult1 = !alwaysTrue
        let notResult2 = !alwaysFalse
        let notResult3 = !mostlyTrue

        #expect(notResult1.sample() == false)
        #expect(notResult2.sample() == true)
        #expect(type(of: notResult3) == Uncertain<Bool>.self)

        // Test logical combinations with evidence
        let complexLogic = (mostlyTrue && !mostlyFalse) || alwaysFalse
        let logicResult = complexLogic.probability(exceeds: 0.5)

        #expect(type(of: logicResult) == Bool.self)
    }

    @Test("Improved SPRT with configurable parameters")
    func testImprovedSPRTWithConfigurableParameters() {
        // Create a distribution with known probability.
        // A fixed seed keeps the decisions the same on every run.
        let rng = SplitMix64(seed: 1)
        let biasedCoin = Uncertain<Double>.bernoulli(probability: 0.7, using: rng)

        // Test with different batch sizes
        let result1 = biasedCoin.evaluateHypothesis(
            threshold: 0.6,
            confidenceLevel: 0.95,
            maxSamples: 1000,
            batchSize: 1
        )

        let result2 = biasedCoin.evaluateHypothesis(
            threshold: 0.6,
            confidenceLevel: 0.95,
            maxSamples: 1000,
            batchSize: 20
        )

        // Both should make the same decision (true probability 0.7 > 0.6)
        #expect(result1.decision == true)
        #expect(result2.decision == true)

        // Test with custom alpha and beta
        let customResult = biasedCoin.evaluateHypothesis(
            threshold: 0.6,
            confidenceLevel: 0.95,
            maxSamples: 1000,
            alpha: 0.01,
            beta: 0.05
        )

        #expect(customResult.decision == true)
        #expect(customResult.samplesUsed > 0)
        #expect(customResult.probability > 0.6)

        // Test with different epsilon values
        let tightEpsilon = biasedCoin.evaluateHypothesis(
            threshold: 0.6,
            confidenceLevel: 0.95,
            maxSamples: 1000,
            epsilon: 0.01
        )

        let looseEpsilon = biasedCoin.evaluateHypothesis(
            threshold: 0.6,
            confidenceLevel: 0.95,
            maxSamples: 1000,
            epsilon: 0.1
        )

        #expect(tightEpsilon.decision == true)
        #expect(looseEpsilon.decision == true)

        // Tight epsilon might require more samples
        #expect(tightEpsilon.samplesUsed >= looseEpsilon.samplesUsed)
    }

    /// Parameters for one SPRT error-rate case.
    struct SPRTErrorRateCase: Sendable, CustomTestStringConvertible {
        var confidenceLevel: Double
        var alpha: Double?
        var beta: Double?

        /// The type I error rate that `evaluateHypothesis` targets.
        var nominalAlpha: Double { alpha ?? (1.0 - confidenceLevel) }
        /// The type II error rate that `evaluateHypothesis` targets.
        var nominalBeta: Double { beta ?? nominalAlpha }

        var testDescription: String {
            if let alpha, let beta {
                return "alpha: \(alpha), beta: \(beta)"
            }
            return "confidenceLevel: \(confidenceLevel)"
        }
    }

    // Measured with the seeds below (2,000 runs for each hypothesis, maxSamples 1,000):
    //
    // | Case                  | Type I rate (alpha) | Type II rate (beta) |
    // | --------------------- | ------------------- | ------------------- |
    // | confidenceLevel 0.90  | 0.0705 (0.10)       | 0.0800 (0.10)       |
    // | confidenceLevel 0.95  | 0.0335 (0.05)       | 0.0315 (0.05)       |
    // | alpha 0.01, beta 0.10 | 0.0060 (0.01)       | 0.0845 (0.10)       |
    //
    // All rates are below their bounds, by 0.004 to 0.030,
    // because each batch of 10 samples overshoots the SPRT boundary,
    // which makes the test more conservative than Wald's approximation.
    // No run reached the fallback decision at maxSamples.
    @Test(
        "SPRT error rates stay near alpha and beta",
        arguments: [
            SPRTErrorRateCase(confidenceLevel: 0.90),
            SPRTErrorRateCase(confidenceLevel: 0.95),
            SPRTErrorRateCase(confidenceLevel: 0.95, alpha: 0.01, beta: 0.10),
        ]
    )
    func testSPRTErrorRates(_ testCase: SPRTErrorRateCase) {
        let threshold = 0.5
        let epsilon = 0.05
        let runs = 2000

        var falsePositives = 0
        var falseNegatives = 0
        for run in 0 ..< runs {
            // Each run uses its own seed, and the two hypotheses use separate seed ranges.
            let atNull = Uncertain<Double>.bernoulli(
                probability: threshold - epsilon,
                using: SplitMix64(seed: UInt64(run))
            )
            let atAlternative = Uncertain<Double>.bernoulli(
                probability: threshold + epsilon,
                using: SplitMix64(seed: UInt64(runs + run))
            )

            if atNull.evaluateHypothesis(
                threshold: threshold,
                confidenceLevel: testCase.confidenceLevel,
                maxSamples: 1000,
                epsilon: epsilon,
                alpha: testCase.alpha,
                beta: testCase.beta
            ).decision {
                falsePositives += 1
            }
            if !atAlternative.evaluateHypothesis(
                threshold: threshold,
                confidenceLevel: testCase.confidenceLevel,
                maxSamples: 1000,
                epsilon: epsilon,
                alpha: testCase.alpha,
                beta: testCase.beta
            ).decision {
                falseNegatives += 1
            }
        }

        // The tolerance is three standard errors of a binomial rate at the nominal value
        // over `runs` trials: about 0.020 for 0.10, 0.015 for 0.05, and 0.0067 for 0.01.
        // The seeds make the result deterministic, so this margin does not guard against chance.
        // It allows a change to the sampling or batching that moves the rates
        // by an amount that a correct SPRT could also show,
        // and it fails when a rate is well above its target.
        func tolerance(_ rate: Double) -> Double {
            3 * (rate * (1 - rate) / Double(runs)).squareRoot()
        }

        let typeIRate = Double(falsePositives) / Double(runs)
        let typeIIRate = Double(falseNegatives) / Double(runs)
        #expect(typeIRate <= testCase.nominalAlpha + tolerance(testCase.nominalAlpha))
        #expect(typeIIRate <= testCase.nominalBeta + tolerance(testCase.nominalBeta))
    }

    @Test("Complex shared variable expressions")
    func testComplexSharedVariableExpressions() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Test complex expressions with shared variables
        let a = Uncertain<Double>.normal(mean: 1.0, standardDeviation: 0.1, using: rng)
        let b = Uncertain<Double>.normal(mean: 2.0, standardDeviation: 0.1, using: rng)

        // Test that (a + b) * (a + b) behaves like (a + b)²
        let sumAB = a + b
        let squared1 = sumAB * sumAB
        let squared2 = (a + b) * (a + b)

        let mean1 = squared1.expectedValue(sampleCount: 1000)
        let mean2 = squared2.expectedValue(sampleCount: 1000)

        // Both should be approximately (1 + 2)² = 9
        #expect(abs(mean1 - 9.0) < 0.5)
        #expect(abs(mean2 - 9.0) < 0.5)
        #expect(abs(mean1 - mean2) < 0.2)

        // Test mixed expression: a * a + b * b
        let mixed = a * a + b * b
        let mixedMean = mixed.expectedValue(sampleCount: 1000)

        // Should be approximately 1² + 2² = 5
        #expect(abs(mixedMean - 5.0) < 0.5)
    }

    @Test("Logical operators preserve shared variable semantics")
    func testLogicalOperatorsPreserveSharedVariableSemantics() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)

        // Test that logical operators properly handle shared variables
        let x = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0, using: rng)
        let threshold = 5.0

        let above = x > threshold
        let below = x < threshold

        // These should be mutually exclusive for the same sample
        let andResult = above && below
        let orResult = above || below

        // AND should be mostly false (can't be both above and below)
        let andProbability = andResult.probability(exceeds: 0.5)
        #expect(andProbability == false)

        // OR should be mostly true (must be either above or below, rarely exactly equal)
        let orProbability = orResult.probability(exceeds: 0.5)
        #expect(orProbability == true)

        // Test De Morgan's laws with shared variables
        let notAndResult = !(above && below)
        let notOrResult = (!above) || (!below)

        // These should be equivalent by De Morgan's law
        let notAndProb = notAndResult.probability(exceeds: 0.5)
        let notOrProb = notOrResult.probability(exceeds: 0.5)

        #expect(notAndProb == notOrProb)
    }

    // MARK: - Shared Samples in Comparisons

    /// Checks that every sample of the evidence has the expected value.
    func expectAlways(
        _ expected: Bool,
        _ evidence: Uncertain<Bool>,
        sampleCount: Int = 1000,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let samples = Array(evidence.prefix(sampleCount))
        #expect(samples.allSatisfy { $0 == expected }, sourceLocation: sourceLocation)
    }

    @Test("Comparisons of a variable with itself use one sample")
    func testComparisonsWithSelfShareSamples() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)
        let x = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 1.0, using: rng)

        expectAlways(true, x == x)
        expectAlways(false, x != x)
        expectAlways(false, x < x)
        expectAlways(false, x > x)
    }

    @Test("Comparisons of mixed expressions share samples")
    func testComparisonsOfMixedExpressionsShareSamples() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)
        let x = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 1.0, using: rng)
        let y = Uncertain<Double>.uniform(min: 1.0, max: 2.0, using: rng)

        expectAlways(true, (x + y) > x)
        expectAlways(true, x < (x + y))
        expectAlways(false, (x + y) == x)
        expectAlways(true, (x - y) != x)
        expectAlways(false, (x > y) && (y > x))
    }

    @Test("Logical operators share samples across comparisons with constants")
    func testLogicalOperatorsShareSamplesWithConstants() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)
        let x = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 1.0, using: rng)
        let above = x > 0.0

        expectAlways(false, above && (x <= 0.0))
        expectAlways(true, above || (x <= 0.0))
        expectAlways(false, (x >= 0.0) && (x < 0.0))
        expectAlways(true, above == !(x <= 0.0))
        expectAlways(false, above && !above)
        expectAlways(true, above || !above)

        // A discrete variable takes each value often, so equality is not rare
        let coin = Uncertain<Int>.binomial(trials: 1, probability: 0.5, using: rng)
        expectAlways(false, (coin == 1) && (coin != 1))
        expectAlways(true, (coin == 0) || (coin != 0))
        expectAlways(true, coin == coin)
    }

    @Test("Separate variables stay independent")
    func testSeparateVariablesStayIndependent() {
        // A fixed seed draws the same samples on every run
        let rng = SplitMix64(seed: 1)
        let x = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 1.0, using: rng)
        let y = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 1.0, using: rng)

        // For independent x and y, P(x < y) is 0.5
        let samples = Array((x < y).prefix(1000))
        let fraction = Double(samples.filter { $0 }.count) / Double(samples.count)
        #expect(abs(fraction - 0.5) < 0.1)
    }
}

#if canImport(CoreLocation)
    import CoreLocation

    @Suite("Uncertain Core Location Extensions")
    struct UncertainCoreLocationTests {
        @Test("Uncertain<CLLocation> from CLLocation models uncertainty")
        func testUncertainCLLocationFrom() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let base = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
                altitude: 10.0,
                horizontalAccuracy: 5.0,
                verticalAccuracy: 2.0,
                timestamp: Date()
            )
            let uncertain = Uncertain<CLLocation>.from(base, using: rng)
            // Should sample near the base location
            let samples = (0 ..< 1000).map { _ in uncertain.sample() }
            let meanLat = samples.map { $0.coordinate.latitude }.reduce(0, +) / 1000
            let meanLon = samples.map { $0.coordinate.longitude }.reduce(0, +) / 1000
            let meanAlt = samples.map { $0.altitude }.reduce(0, +) / 1000
            #expect(abs(meanLat - base.coordinate.latitude) < 0.01)
            #expect(abs(meanLon - base.coordinate.longitude) < 0.01)
            #expect(abs(meanAlt - base.altitude) < 0.5)
        }

        @Test("Uncertain<CLLocation> .location constructor")
        func testUncertainCLLocationLocation() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let coord = CLLocationCoordinate2D(latitude: 40.0, longitude: -74.0)
            let uncertain = Uncertain<CLLocation>.location(
                coordinate: coord,
                horizontalAccuracy: 10.0,
                verticalAccuracy: 3.0,
                altitude: 50.0,
                using: rng
            )
            let samples = (0 ..< 1000).map { _ in uncertain.sample() }
            let meanLat = samples.map { $0.coordinate.latitude }.reduce(0, +) / 1000
            let meanLon = samples.map { $0.coordinate.longitude }.reduce(0, +) / 1000
            let meanAlt = samples.map { $0.altitude }.reduce(0, +) / 1000
            #expect(abs(meanLat - coord.latitude) < 0.02)
            #expect(abs(meanLon - coord.longitude) < 0.02)
            #expect(abs(meanAlt - 50.0) < 1.0)
        }

        @Test("Uncertain<CLLocation> distance and bearing")
        func testUncertainCLLocationDistanceAndBearing() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let a = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
                horizontalAccuracy: 5.0,
                using: rng
            )
            let b = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7750, longitude: -122.4184),
                horizontalAccuracy: 5.0,
                using: rng
            )
            let distance = Uncertain<CLLocation>.distance(from: a, to: b)
            let bearing = Uncertain<CLLocation>.bearing(from: a, to: b)
            let meanDistance = (0 ..< 1000).map { _ in distance.sample() }.reduce(0, +) / 1000
            let meanBearing = (0 ..< 1000).map { _ in bearing.sample() }.reduce(0, +) / 1000
            // Should be close to the true distance and bearing
            let trueA = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
                altitude: 0,
                horizontalAccuracy: 1.0,
                verticalAccuracy: 1.0,
                timestamp: Date()
            )
            let trueB = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 37.7750, longitude: -122.4184),
                altitude: 0,
                horizontalAccuracy: 1.0,
                verticalAccuracy: 1.0,
                timestamp: Date()
            )
            let trueDistance = trueA.distance(from: trueB)
            #expect(abs(meanDistance - trueDistance) < 10.0)
            #expect(meanBearing >= 0 && meanBearing <= 360)
        }

        @Test("Uncertain<CLLocationCoordinate2D> coordinate and distance")
        func testUncertainCLLocationCoordinate2D() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let coordA = CLLocationCoordinate2D(latitude: 51.5, longitude: -0.1)
            let coordB = CLLocationCoordinate2D(latitude: 51.5005, longitude: -0.099)
            let a = Uncertain<CLLocationCoordinate2D>.coordinate(coordA, accuracy: 8.0, using: rng)
            let b = Uncertain<CLLocationCoordinate2D>.coordinate(coordB, accuracy: 8.0, using: rng)
            let distance = Uncertain<CLLocationCoordinate2D>.distance(from: a, to: b)
            let meanDistance = (0 ..< 1000).map { _ in distance.sample() }.reduce(0, +) / 1000
            let trueA = CLLocation(
                coordinate: coordA,
                altitude: 0,
                horizontalAccuracy: 1.0,
                verticalAccuracy: 1.0,
                timestamp: Date()
            )
            let trueB = CLLocation(
                coordinate: coordB,
                altitude: 0,
                horizontalAccuracy: 1.0,
                verticalAccuracy: 1.0,
                timestamp: Date()
            )
            let trueDistance = trueA.distance(from: trueB)
            #expect(abs(meanDistance - trueDistance) < 15.0)
        }

        @Test("Uncertain<CLLocationSpeed> from location")
        func testUncertainCLLocationSpeed() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let loc = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                altitude: 0,
                horizontalAccuracy: 10.0,
                verticalAccuracy: 5.0,
                course: 90.0,
                speed: 15.0,
                timestamp: Date()
            )
            if #available(iOS 14.0, macOS 11.0, *) {
                let speed = Uncertain<CLLocationSpeed>.speed(from: loc, using: rng)
                let samples = (0 ..< 1000).map { _ in speed.sample() }
                let meanSpeed = samples.reduce(0, +) / 1000
                #expect(abs(meanSpeed - 15.0) < 1.0)
            }
        }

        @Test("Uncertain<CLLocationDirection> from location")
        func testUncertainCLLocationDirection() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let loc = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                altitude: 0,
                horizontalAccuracy: 10.0,
                verticalAccuracy: 5.0,
                course: 45.0,
                speed: 10.0,
                timestamp: Date()
            )
            if #available(iOS 14.0, macOS 11.0, *) {
                let course = Uncertain<CLLocationDirection>.course(from: loc, using: rng)
                let samples = (0 ..< 1000).map { _ in course.sample() }
                let meanCourse = samples.reduce(0, +) / 1000
                #expect(abs(meanCourse - 45.0) < 5.0)
                #expect(meanCourse >= 0 && meanCourse <= 360)
            }
        }

        @Test("Parameterized speed uncertainty")
        func testParameterizedSpeedUncertainty() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let loc = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                altitude: 0,
                horizontalAccuracy: 20.0,
                verticalAccuracy: 5.0,
                course: 90.0,
                speed: 10.0,
                timestamp: Date()
            )
            if #available(iOS 14.0, macOS 11.0, *) {
                // Test with different uncertainty parameters
                let conservativeSpeed = Uncertain<CLLocationSpeed>.speed(
                    from: loc,
                    speedUncertaintyFactor: 0.05,
                    minimumSpeedUncertainty: 0.5,
                    using: rng
                )
                let liberalSpeed = Uncertain<CLLocationSpeed>.speed(
                    from: loc,
                    speedUncertaintyFactor: 0.2,
                    minimumSpeedUncertainty: 0.1,
                    using: rng
                )

                let conservativeSamples = (0 ..< 1000).map { _ in conservativeSpeed.sample() }
                let liberalSamples = (0 ..< 1000).map { _ in liberalSpeed.sample() }

                let conservativeStd = sqrt(
                    conservativeSamples.map { pow($0 - 10.0, 2) }.reduce(0, +) / 1000
                )
                let liberalStd = sqrt(liberalSamples.map { pow($0 - 10.0, 2) }.reduce(0, +) / 1000)

                // Liberal should have higher uncertainty
                #expect(liberalStd > conservativeStd)
            }
        }

        @Test("Parameterized course uncertainty")
        func testParameterizedCourseUncertainty() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let loc = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                altitude: 0,
                horizontalAccuracy: 10.0,
                verticalAccuracy: 5.0,
                course: 90.0,
                speed: 10.0,
                timestamp: Date()
            )
            if #available(iOS 14.0, macOS 11.0, *) {
                // Test with different course uncertainties
                let preciseCourse = Uncertain<CLLocationDirection>.course(
                    from: loc,
                    courseAccuracy: 2.0,
                    using: rng
                )
                let roughCourse = Uncertain<CLLocationDirection>.course(
                    from: loc,
                    courseAccuracy: 15.0,
                    using: rng
                )

                let preciseSamples = (0 ..< 1000).map { _ in preciseCourse.sample() }
                let roughSamples = (0 ..< 1000).map { _ in roughCourse.sample() }

                // Both should be centered around 90 degrees
                let preciseMean = preciseSamples.reduce(0, +) / 1000
                let roughMean = roughSamples.reduce(0, +) / 1000

                #expect(abs(preciseMean - 90.0) < 2.0)
                #expect(abs(roughMean - 90.0) < 10.0)

                // Rough should have higher variance
                let preciseVar = preciseSamples.map { pow($0 - 90.0, 2) }.reduce(0, +) / 1000
                let roughVar = roughSamples.map { pow($0 - 90.0, 2) }.reduce(0, +) / 1000

                #expect(roughVar > preciseVar)
            }
        }

        @Test("Course uncertainty from location courseAccuracy")
        func testCourseUncertaintyFromLocationAccuracy() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            if #available(iOS 14.0, macOS 11.0, *) {
                // Create location with specific courseAccuracy
                let locWithAccurateCourse = CLLocation(
                    coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                    altitude: 0,
                    horizontalAccuracy: 10.0,
                    verticalAccuracy: 5.0,
                    course: 180.0,  // South
                    courseAccuracy: 2.0,  // High accuracy
                    speed: 10.0,
                    speedAccuracy: 1.0,
                    timestamp: Date()
                )

                let locWithRoughCourse = CLLocation(
                    coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                    altitude: 0,
                    horizontalAccuracy: 10.0,
                    verticalAccuracy: 5.0,
                    course: 180.0,  // South
                    courseAccuracy: 20.0,  // Low accuracy
                    speed: 10.0,
                    speedAccuracy: 1.0,
                    timestamp: Date()
                )

                // Test that the library uses the location's built-in courseAccuracy
                let accurateCourse = Uncertain<CLLocationDirection>.course(
                    from: locWithAccurateCourse,
                    using: rng
                )
                let roughCourse = Uncertain<CLLocationDirection>.course(from: locWithRoughCourse, using: rng)

                let accurateSamples = (0 ..< 1000).map { _ in accurateCourse.sample() }
                let roughSamples = (0 ..< 1000).map { _ in roughCourse.sample() }

                // Both should be centered around 180 degrees
                let accurateMean = accurateSamples.reduce(0, +) / 1000
                let roughMean = roughSamples.reduce(0, +) / 1000

                #expect(abs(accurateMean - 180.0) < 5.0)
                #expect(abs(roughMean - 180.0) < 15.0)

                // Rough course should have higher variance than accurate course
                let accurateVar = accurateSamples.map { pow($0 - 180.0, 2) }.reduce(0, +) / 1000
                let roughVar = roughSamples.map { pow($0 - 180.0, 2) }.reduce(0, +) / 1000

                #expect(roughVar > accurateVar * 2)  // Should be significantly more variable

                // Test invalid course handling
                let locWithInvalidCourse = CLLocation(
                    coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                    altitude: 0,
                    horizontalAccuracy: 10.0,
                    verticalAccuracy: 5.0,
                    course: -1.0,  // Invalid course
                    courseAccuracy: 5.0,
                    speed: 10.0,
                    speedAccuracy: 1.0,
                    timestamp: Date()
                )

                let invalidCourse = Uncertain<CLLocationDirection>.course(
                    from: locWithInvalidCourse,
                    using: rng
                )
                let invalidSample = invalidCourse.sample()
                #expect(invalidSample == -1.0)  // Should return invalid course unchanged

                // Test invalid courseAccuracy handling
                let locWithInvalidAccuracy = CLLocation(
                    coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                    altitude: 0,
                    horizontalAccuracy: 10.0,
                    verticalAccuracy: 5.0,
                    course: 90.0,  // Valid course
                    courseAccuracy: -1.0,  // Invalid accuracy
                    speed: 10.0,
                    speedAccuracy: 1.0,
                    timestamp: Date()
                )

                let invalidAccuracyCourse = Uncertain<CLLocationDirection>.course(
                    from: locWithInvalidAccuracy,
                    using: rng
                )
                let invalidAccuracySample = invalidAccuracyCourse.sample()
                #expect(invalidAccuracySample == 90.0)  // Should return base course unchanged
            }
        }

        @Test("Invalid location data handling")
        func testInvalidLocationDataHandling() {
            // Test with invalid accuracy
            let invalidLoc = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                altitude: 0,
                horizontalAccuracy: -1.0,  // Invalid
                verticalAccuracy: -1.0,  // Invalid
                course: -1.0,  // Invalid
                speed: -1.0,  // Invalid
                timestamp: Date()
            )

            let uncertainLoc = Uncertain<CLLocation>.from(invalidLoc)
            let sample = uncertainLoc.sample()

            // Should return original location when accuracy is invalid
            #expect(sample.coordinate.latitude == invalidLoc.coordinate.latitude)
            #expect(sample.coordinate.longitude == invalidLoc.coordinate.longitude)

            if #available(iOS 14.0, macOS 11.0, *) {
                // Speed and course should return -1 for invalid data
                let speed = Uncertain<CLLocationSpeed>.speed(from: invalidLoc)
                let course = Uncertain<CLLocationDirection>.course(from: invalidLoc)

                #expect(speed.sample() == -1.0)
                #expect(course.sample() == -1.0)
            }
        }

        @Test("GPS uncertainty modeling vs naive approach")
        func testGPSUncertaintyModelingVsNaive() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let baseCoord = CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194)
            let accuracy = 10.0

            // Test the improved coordinate method
            let uncertainCoord = Uncertain<CLLocationCoordinate2D>.coordinate(
                baseCoord,
                accuracy: accuracy,
                using: rng
            )

            // Sample many times to check distribution
            let samples = (0 ..< 1000).map { _ in uncertainCoord.sample() }
            let latOffsets = samples.map { $0.latitude - baseCoord.latitude }
            let lonOffsets = samples.map { $0.longitude - baseCoord.longitude }

            // Mean should be close to base coordinate
            let meanLatOffset = latOffsets.reduce(0, +) / 1000
            let meanLonOffset = lonOffsets.reduce(0, +) / 1000

            #expect(abs(meanLatOffset) < 0.001)
            #expect(abs(meanLonOffset) < 0.001)

            // Should have reasonable distribution (not too tight, not too loose)
            let latStd = sqrt(latOffsets.map { pow($0, 2) }.reduce(0, +) / 1000)
            let lonStd = sqrt(lonOffsets.map { pow($0, 2) }.reduce(0, +) / 1000)

            #expect(latStd > 0.00005)  // Not too tight
            #expect(latStd < 0.01)  // Not too loose
            #expect(lonStd > 0.00005)
            #expect(lonStd < 0.01)
        }

        @Test("Altitude uncertainty modeling")
        func testAltitudeUncertaintyModeling() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let baseLoc = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                altitude: 100.0,
                horizontalAccuracy: 5.0,
                verticalAccuracy: 3.0,
                timestamp: Date()
            )

            let uncertainLoc = Uncertain<CLLocation>.from(baseLoc, using: rng)
            let samples = (0 ..< 1000).map { _ in uncertainLoc.sample() }
            let altitudes = samples.map { $0.altitude }

            // Mean should be close to base altitude
            let meanAlt = altitudes.reduce(0, +) / 1000
            #expect(abs(meanAlt - 100.0) < 1.0)

            // Should have reasonable altitude uncertainty
            let altStd = sqrt(altitudes.map { pow($0 - 100.0, 2) }.reduce(0, +) / 1000)
            #expect(altStd > 1.0)  // Should have some uncertainty
            #expect(altStd < 10.0)  // But not too much
        }

        @Test("Location uncertainty propagation in calculations")
        func testLocationUncertaintyPropagationInCalculations() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let locA = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
                horizontalAccuracy: 10.0,
                altitude: 50.0,
                using: rng
            )

            let locB = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7849, longitude: -122.4094),
                horizontalAccuracy: 10.0,
                altitude: 60.0,
                using: rng
            )

            // Calculate uncertain distance and bearing
            let distance = Uncertain<CLLocation>.distance(from: locA, to: locB)
            let bearing = Uncertain<CLLocation>.bearing(from: locA, to: locB)

            // Test that uncertainty propagates correctly
            let distanceSamples = (0 ..< 1000).map { _ in distance.sample() }
            let bearingSamples = (0 ..< 1000).map { _ in bearing.sample() }

            // Distance should be around 1.5km (rough estimate)
            let meanDistance = distanceSamples.reduce(0, +) / 1000
            #expect(meanDistance > 1000)  // > 1km
            #expect(meanDistance < 3000)  // < 3km

            // Should have reasonable uncertainty in distance
            let distanceStd = sqrt(
                distanceSamples.map { pow($0 - meanDistance, 2) }.reduce(0, +) / 1000
            )
            #expect(distanceStd > 5.0)  // Some uncertainty
            #expect(distanceStd < 50.0)  // Not too much

            // Bearing should be roughly northeast
            let meanBearing = bearingSamples.reduce(0, +) / 1000
            #expect(meanBearing >= 0)
            #expect(meanBearing <= 360)
        }

        @Test("Course angle normalization")
        func testCourseAngleNormalization() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            if #available(iOS 14.0, macOS 11.0, *) {
                // Test course near 0/360 boundary
                let locNorth = CLLocation(
                    coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                    altitude: 0,
                    horizontalAccuracy: 5.0,
                    verticalAccuracy: 5.0,
                    course: 359.0,  // Very close to north
                    speed: 10.0,
                    timestamp: Date()
                )

                let course = Uncertain<CLLocationDirection>.course(
                    from: locNorth,
                    courseAccuracy: 10.0,  // Large uncertainty to test boundary
                    using: rng
                )

                let samples = (0 ..< 1000).map { _ in course.sample() }

                // All samples should be in valid range [0, 360)
                #expect(samples.allSatisfy { $0 >= 0 && $0 < 360 })

                // Test with course at exact boundary
                let locExact = CLLocation(
                    coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                    altitude: 0,
                    horizontalAccuracy: 5.0,
                    verticalAccuracy: 5.0,
                    course: 0.0,
                    speed: 10.0,
                    timestamp: Date()
                )

                let exactCourse = Uncertain<CLLocationDirection>.course(from: locExact, using: rng)
                let exactSamples = (0 ..< 100).map { _ in exactCourse.sample() }

                #expect(exactSamples.allSatisfy { $0 >= 0 && $0 < 360 })
            }
        }

        @Test("CLLocation instance methods for distance calculations")
        func testCLLocationInstanceMethods() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let locA = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
                horizontalAccuracy: 5.0,
                using: rng
            )
            let locB = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7849, longitude: -122.4094),
                horizontalAccuracy: 5.0,
                using: rng
            )
            let specificLoc = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 37.7849, longitude: -122.4094),
                altitude: 0,
                horizontalAccuracy: 5.0,
                verticalAccuracy: 5.0,
                timestamp: Date()
            )

            // Test instance methods vs static methods
            let instanceDistance1 = locA.distance(to: locB)
            let instanceDistance2 = locA.distance(to: specificLoc)
            let staticDistance = Uncertain<CLLocation>.distance(from: locA, to: locB)

            let instanceBearing1 = locA.bearing(to: locB)
            let instanceBearing2 = locA.bearing(to: specificLoc)
            let staticBearing = Uncertain<CLLocation>.bearing(from: locA, to: locB)

            // Compare means to ensure equivalence
            let instanceDist1Mean =
                (0 ..< 100).map { _ in instanceDistance1.sample() }.reduce(0, +) / 100
            let instanceDist2Mean =
                (0 ..< 100).map { _ in instanceDistance2.sample() }.reduce(0, +) / 100
            let staticDistMean = (0 ..< 100).map { _ in staticDistance.sample() }.reduce(0, +) / 100

            let instanceBear1Mean =
                (0 ..< 100).map { _ in instanceBearing1.sample() }.reduce(0, +) / 100
            let instanceBear2Mean =
                (0 ..< 100).map { _ in instanceBearing2.sample() }.reduce(0, +) / 100
            let staticBearMean = (0 ..< 100).map { _ in staticBearing.sample() }.reduce(0, +) / 100

            // Instance and static methods should be equivalent
            #expect(abs(instanceDist1Mean - staticDistMean) < 50)
            #expect(abs(instanceBear1Mean - staticBearMean) < 10)

            // Distance to specific location should be reasonable
            #expect(instanceDist2Mean > 500)  // Should be some distance
            #expect(instanceDist2Mean < 3000)  // But not too far

            // Bearing to specific location should be valid
            #expect(instanceBear2Mean >= 0)
            #expect(instanceBear2Mean <= 360)
        }

        @Test("CLLocationCoordinate2D instance methods")
        func testCLLocationCoordinate2DInstanceMethods() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            let coordA = Uncertain<CLLocationCoordinate2D>.coordinate(
                CLLocationCoordinate2D(latitude: 40.7128, longitude: -74.0060),
                accuracy: 8.0,
                using: rng
            )
            let coordB = Uncertain<CLLocationCoordinate2D>.coordinate(
                CLLocationCoordinate2D(latitude: 40.7589, longitude: -73.9851),
                accuracy: 8.0,
                using: rng
            )
            let specificCoord = CLLocationCoordinate2D(latitude: 40.7589, longitude: -73.9851)

            // Test distance methods
            let instanceDistance1 = coordA.distance(to: coordB)
            let instanceDistance2 = coordA.distance(to: specificCoord)
            let staticDistance = Uncertain<CLLocationCoordinate2D>.distance(
                from: coordA,
                to: coordB
            )

            // Test bearing methods
            let instanceBearing1 = coordA.bearing(to: coordB)
            let instanceBearing2 = coordA.bearing(to: specificCoord)
            let staticBearing = Uncertain<CLLocationCoordinate2D>.bearing(from: coordA, to: coordB)

            // Sample to check equivalence
            let instanceDist1Mean =
                (0 ..< 100).map { _ in instanceDistance1.sample() }.reduce(0, +) / 100
            let instanceDist2Mean =
                (0 ..< 100).map { _ in instanceDistance2.sample() }.reduce(0, +) / 100
            let staticDistMean = (0 ..< 100).map { _ in staticDistance.sample() }.reduce(0, +) / 100

            let instanceBear1Mean =
                (0 ..< 100).map { _ in instanceBearing1.sample() }.reduce(0, +) / 100
            let instanceBear2Mean =
                (0 ..< 100).map { _ in instanceBearing2.sample() }.reduce(0, +) / 100
            let staticBearMean = (0 ..< 100).map { _ in staticBearing.sample() }.reduce(0, +) / 100

            // Should be equivalent to static methods
            #expect(abs(instanceDist1Mean - staticDistMean) < 100)
            #expect(abs(instanceBear1Mean - staticBearMean) < 15)

            // Distance between NYC coordinates should be reasonable (~5-6km)
            #expect(instanceDist2Mean > 3000)
            #expect(instanceDist2Mean < 8000)

            // Bearing should be valid
            #expect(instanceBear2Mean >= 0)
            #expect(instanceBear2Mean <= 360)
        }

        @Test("CLLocationSpeed instance methods for unit conversions")
        func testCLLocationSpeedInstanceMethods() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            if #available(iOS 14.0, macOS 11.0, *) {
                let baseSpeed = 20.0  // 20 m/s
                let location = CLLocation(
                    coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                    altitude: 0,
                    horizontalAccuracy: 5.0,
                    verticalAccuracy: 5.0,
                    course: 90.0,
                    speed: baseSpeed,
                    timestamp: Date()
                )

                let uncertainSpeed = Uncertain<CLLocationSpeed>.speed(from: location, using: rng)

                // Test speed comparison methods
                let speedLimit = 15.0  // m/s
                let exceedsLimit = uncertainSpeed.exceeds(speedLimit)
                let withinRange = uncertainSpeed.isWithin(min: 18.0, max: 22.0)

                // With base speed 20 m/s, should exceed 15 m/s limit
                let exceedsResult = exceedsLimit.probability(exceeds: 0.5)
                #expect(exceedsResult == true)

                // Should be within 18-22 m/s range
                let withinResult = withinRange.probability(exceeds: 0.5)
                #expect(withinResult == true)

                // Test evidence-based evaluation
                let definitelyFast = exceedsLimit.probability(exceeds: 0.9)
                #expect(definitelyFast == true)
            }
        }

        @Test("CLLocationDirection instance methods for angular calculations")
        func testCLLocationDirectionInstanceMethods() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            if #available(iOS 14.0, macOS 11.0, *) {
                let baseCourse = 90.0  // Due east
                let location = CLLocation(
                    coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                    altitude: 0,
                    horizontalAccuracy: 5.0,
                    verticalAccuracy: 5.0,
                    course: baseCourse,
                    speed: 10.0,
                    timestamp: Date()
                )

                let uncertainDirection = Uncertain<CLLocationDirection>.course(from: location, using: rng)

                // Test angular difference calculations
                let targetDirection1 = 45.0  // 45° difference
                let targetDirection2 = 135.0  // 45° difference (other direction)

                let diff1 = uncertainDirection.angularDifference(to: targetDirection1)
                let diff2 = uncertainDirection.angularDifference(to: targetDirection2)

                let diff1Mean = (0 ..< 100).map { _ in diff1.sample() }.reduce(0, +) / 100
                let diff2Mean = (0 ..< 100).map { _ in diff2.sample() }.reduce(0, +) / 100

                // Expected differences: -45°, +45°, +180°
                #expect(abs(diff1Mean - (-45.0)) < 10)
                #expect(abs(diff2Mean - 45.0) < 10)

                // Test angular difference with uncertain direction
                let targetUncertain = Uncertain<CLLocationDirection>.course(
                    from: CLLocation(
                        coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                        altitude: 0,
                        horizontalAccuracy: 5.0,
                        verticalAccuracy: 5.0,
                        course: 45.0,
                        speed: 10.0,
                        timestamp: Date()
                    ),
                    using: rng
                )

                let uncertainDiff = uncertainDirection.angularDifference(to: targetUncertain)
                let uncertainDiffMean =
                    (0 ..< 100).map { _ in uncertainDiff.sample() }.reduce(0, +) / 100

                // Should be approximately -45° (90° - 45°)
                #expect(abs(uncertainDiffMean - (-45.0)) < 15)

                // Test within tolerance check
                let withinTolerance = uncertainDirection.isWithin(10.0, of: 85.0)
                let toleranceResult = withinTolerance.probability(exceeds: 0.5)

                // 90° should be within 10° of 85°
                #expect(toleranceResult == true)

                // Test edge case around 0/360 boundary
                let northDirection = Uncertain<CLLocationDirection>.course(
                    from: CLLocation(
                        coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                        altitude: 0,
                        horizontalAccuracy: 5.0,
                        verticalAccuracy: 5.0,
                        course: 0.0,
                        speed: 10.0,
                        timestamp: Date()
                    ),
                    courseAccuracy: 2.0,
                    using: rng
                )

                let boundaryDiff = northDirection.angularDifference(to: 350.0)
                let boundaryDiffMean =
                    (0 ..< 100).map { _ in boundaryDiff.sample() }.reduce(0, +) / 100

                // 0° to 350° should be -10° (shortest path)
                #expect(abs(boundaryDiffMean - (-10.0)) < 5)
            }
        }

        @Test("Speed limit enforcement example with instance methods")
        func testSpeedLimitEnforcementWithInstanceMethods() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            if #available(iOS 14.0, macOS 11.0, *) {
                // Simulate a speed enforcement scenario
                let location = CLLocation(
                    coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                    altitude: 0,
                    // 0.5 m/s speed uncertainty, so the speed is clearly in the legal range
                    horizontalAccuracy: 5.0,
                    verticalAccuracy: 5.0,
                    course: 90.0,
                    speed: 28.0,  // 28 m/s ≈ 63 mph
                    timestamp: Date()
                )

                let uncertainSpeed = Uncertain<CLLocationSpeed>.speed(from: location, using: rng)

                // Different speed limits
                let cityLimit = 25.0 * 0.44704  // 25 mph in m/s
                let highwayLimit = 65.0 * 0.44704  // 65 mph in m/s

                // Test evidence-based enforcement
                let exceedsCityLimit = uncertainSpeed.exceeds(cityLimit)
                let exceedsHighwayLimit = uncertainSpeed.exceeds(highwayLimit)

                // Conservative enforcement (95% confidence)
                let conservativeCity = exceedsCityLimit.probability(exceeds: 0.95)
                let conservativeHighway = exceedsHighwayLimit.probability(exceeds: 0.95)

                // Standard enforcement (>50% confidence)
                let standardCity = ~exceedsCityLimit
                let standardHighway = ~exceedsHighwayLimit

                // At 28 m/s (≈63 mph):
                // - Should exceed 25 mph city limit
                // - Should not exceed 65 mph highway limit
                #expect(conservativeCity == true)
                #expect(standardCity == true)
                #expect(conservativeHighway == false)
                #expect(standardHighway == false)

                // Test speed range checking
                let legalHighwayRange = uncertainSpeed.isWithin(
                    min: 20.0 * 0.44704,  // 20 mph minimum
                    max: 65.0 * 0.44704  // 65 mph maximum
                )

                let inLegalRange = legalHighwayRange.probability(exceeds: 0.8)
                #expect(inLegalRange == true)
            }
        }

        @Test("Navigation accuracy with direction instance methods")
        func testNavigationAccuracyWithDirectionInstanceMethods() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            if #available(iOS 14.0, macOS 11.0, *) {
                // Simulate a navigation scenario
                let currentHeading = Uncertain<CLLocationDirection>.course(
                    from: CLLocation(
                        coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                        altitude: 0,
                        horizontalAccuracy: 5.0,
                        verticalAccuracy: 5.0,
                        course: 45.0,  // Northeast
                        speed: 15.0,
                        timestamp: Date()
                    ),
                    // 5° accuracy puts 98% of headings within tolerance
                    courseAccuracy: 5.0,
                    using: rng
                )

                let desiredHeading = 50.0  // Slightly more east
                let acceptableTolerance = 15.0  // ±15 degrees

                // Test if we're on course
                let onCourse = currentHeading.isWithin(acceptableTolerance, of: desiredHeading)
                let navigationAccurate = onCourse.probability(exceeds: 0.8)

                // Should be on course (45° is within 15° of 50°)
                #expect(navigationAccurate == true)

                // Test course correction needed
                let courseCorrection = currentHeading.angularDifference(to: desiredHeading)
                let correctionMean =
                    (0 ..< 200).map { _ in courseCorrection.sample() }.reduce(0, +) / 200

                // Should need about +5° correction (50° - 45°)
                #expect(abs(correctionMean - 5.0) < 3.0)

                // Test with larger course deviation
                let severeDeviation = currentHeading.isWithin(5.0, of: 90.0)
                let severelyOffCourse = severeDeviation.probability(exceeds: 0.5)

                // 45° should NOT be within 5° of 90°
                #expect(severelyOffCourse == false)
            }
        }

        @Test("Complex location calculations with instance methods")
        func testComplexLocationCalculationsWithInstanceMethods() {
            // A fixed seed draws the same samples on every run
            let rng = SplitMix64(seed: 1)

            // Test chaining of instance methods for complex calculations
            let startLocation = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
                horizontalAccuracy: 10.0,
                using: rng
            )

            let checkpoint1 = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 37.7849, longitude: -122.4094),
                altitude: 0,
                horizontalAccuracy: 5.0,
                verticalAccuracy: 5.0,
                timestamp: Date()
            )
            let checkpoint2 = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 37.7949, longitude: -122.3994),
                altitude: 0,
                horizontalAccuracy: 5.0,
                verticalAccuracy: 5.0,
                timestamp: Date()
            )

            // Calculate distances to checkpoints
            let distanceToCheckpoint1 = startLocation.distance(to: checkpoint1)
            let distanceToCheckpoint2 = startLocation.distance(to: checkpoint2)

            // Calculate bearings to checkpoints
            let bearingToCheckpoint1 = startLocation.bearing(to: checkpoint1)
            let bearingToCheckpoint2 = startLocation.bearing(to: checkpoint2)

            // Test evidence-based decisions
            let closeToCheckpoint1 = distanceToCheckpoint1 < 2000.0  // Within 2km
            let closeToCheckpoint2 = distanceToCheckpoint2 < 3000.0  // Within 3km

            let checkpoint1Reachable = closeToCheckpoint1.probability(exceeds: 0.9)
            let checkpoint2Reachable = closeToCheckpoint2.probability(exceeds: 0.9)

            #expect(checkpoint1Reachable == true)
            #expect(checkpoint2Reachable == true)

            // Test bearing calculations are reasonable
            let bearing1Samples = (0 ..< 100).map { _ in bearingToCheckpoint1.sample() }
            let bearing2Samples = (0 ..< 100).map { _ in bearingToCheckpoint2.sample() }

            #expect(bearing1Samples.allSatisfy { $0 >= 0 && $0 <= 360 })
            #expect(bearing2Samples.allSatisfy { $0 >= 0 && $0 <= 360 })

            // Both checkpoints are northeast, so bearings should be in that quadrant
            let bearing1Mean = bearing1Samples.reduce(0, +) / 100
            let bearing2Mean = bearing2Samples.reduce(0, +) / 100

            #expect(bearing1Mean > 0 && bearing1Mean < 90)  // Northeast quadrant
            #expect(bearing2Mean > 0 && bearing2Mean < 90)  // Northeast quadrant
        }
    }
#endif
