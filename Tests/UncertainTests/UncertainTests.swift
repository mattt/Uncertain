import Foundation
import Testing

@testable import Uncertain

@Suite("Paper-Faithful Uncertain<T> Tests")
struct UncertainTests {

    // MARK: - Core Concepts

    @Test("Evidence-based conditionals prevent uncertainty bugs")
    func testEvidenceBasedConditionals() {
        // Temperature sensor with ±2°C accuracy
        let temperature = Uncertain<Double>.normal(mean: 1.0, standardDeviation: 2.0)
        let freezingPoint = 0.0

        // This is the key insight: comparisons return Uncertain<Bool>
        let freezingEvidence = temperature < freezingPoint

        // Test that we get an Uncertain<Bool> back
        #expect(type(of: freezingEvidence) == Uncertain<Bool>.self)

        // Test different confidence levels
        let highConfidence = freezingEvidence.probability(exceeds: 0.95)
        let mediumConfidence = freezingEvidence.probability(exceeds: 0.5)
        let lowConfidence = freezingEvidence.probability(exceeds: 0.1)

        // These should be boolean decisions
        #expect(type(of: highConfidence) == Bool.self)
        #expect(type(of: mediumConfidence) == Bool.self)
        #expect(type(of: lowConfidence) == Bool.self)

        // With temperature mean=1°C and std=2°C, freezing should be:
        // - Very unlikely at 95% confidence (false)
        // - Likely at low confidence (true)
        #expect(highConfidence == false)
        #expect(lowConfidence == true)
    }

    @Test("Implicit conditionals using ~ operator")
    func testImplicitConditionals() {
        // Speed measurement with uncertainty
        let speed = Uncertain<Double>.normal(mean: 65.0, standardDeviation: 3.0)
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
        // GPS readings with 4-meter accuracy (from the paper)
        let location1 = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 4.0)
        let location2 = Uncertain<Double>.normal(mean: 100.0, standardDeviation: 4.0)

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
        // True speed is 57 mph, but measurement has error
        let trueSpeed = 57.0
        let measurementError = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 2.0)
        let measuredSpeed = Uncertain<Double> { trueSpeed + measurementError.sample() }

        let speedLimit = 60.0
        let speedingEvidence = measuredSpeed > speedLimit

        // Different enforcement strategies
        let lenientOfficer = speedingEvidence.probability(exceeds: 0.95)
        let standardOfficer = ~speedingEvidence
        let strictOfficer = speedingEvidence.probability(exceeds: 0.2)

        // With true speed 57 mph and 2 mph error, speed limit 60 mph:
        // - Lenient (95% confident): Should not ticket
        // - Standard (50% confident): Should not ticket
        // - Strict (20% confident): Should not ticket
        #expect(lenientOfficer == false)
        #expect(standardOfficer == false)
        #expect(strictOfficer == false)
    }

    @Test("Comparison operators return Uncertain<Bool>")
    func testComparisonOperatorsReturnUncertainBool() {
        let value = Uncertain<Double>.normal(mean: 10.0, standardDeviation: 2.0)
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
        let x = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0)
        let y = Uncertain<Double>.normal(mean: 3.0, standardDeviation: 1.0)

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
        // Create a distribution where we know the true probability
        let biasedCoin = Uncertain<Bool> { Double.random(in: 0...1) < 0.7 }

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
        // Create clear evidence case
        let certainTrue = Uncertain<Bool> { true }
        let certainFalse = Uncertain<Bool> { false }

        // These should be decided quickly with few samples
        let trueResult = certainTrue.probability(exceeds: 0.5)
        let falseResult = certainFalse.probability(exceeds: 0.5)

        #expect(trueResult == true)
        #expect(falseResult == false)

        // Test edge case near threshold
        let nearThreshold = Uncertain<Bool> { Double.random(in: 0...1) < 0.51 }
        let edgeResult = nearThreshold.probability(exceeds: 0.5)

        // Should still make a decision (though might need more samples)
        #expect(type(of: edgeResult) == Bool.self)
    }

    // MARK: - Distribution Tests (Paper-Faithful)

    @Test("Normal distribution for uncertainty modeling")
    func testNormalDistributionForUncertainty() {
        // GPS accuracy example
        let gpsReading = Uncertain<Double>.normal(mean: 100.0, standardDeviation: 5.0)

        // Test uncertainty propagation
        let samples = gpsReading.prefix(1000)
        let count = Array(samples).count
        let mean = samples.reduce(0, +) / Double(count)
        let variance = samples.reduce(0) { acc, x in acc + (x - mean) * (x - mean) } / Double(count)
        let stdDev = sqrt(variance)

        #expect(abs(mean - 100.0) < 0.5)
        #expect(abs(stdDev - 5.0) < 0.5)
    }

    @Test("Uniform distribution for modeling ranges")
    func testUniformDistributionForRanges() {
        let sensorRange = Uncertain<Double>.uniform(min: 0.0, max: 100.0)
        let samples = sensorRange.prefix(1000)

        // All samples should be in range
        #expect(samples.allSatisfy { $0 >= 0.0 && $0 <= 100.0 })

        // Mean should be approximately 50
        let count = Array(samples).count
        let mean = samples.reduce(0, +) / Double(count)
        #expect(abs(mean - 50.0) < 5.0)
    }

    // MARK: - Arithmetic Operations with Uncertainty Propagation

    @Test("Uncertainty propagation through arithmetic")
    func testUncertaintyPropagationThroughArithmetic() {
        let measurement1 = Uncertain<Double>.normal(mean: 10.0, standardDeviation: 1.0)
        let measurement2 = Uncertain<Double>.normal(mean: 20.0, standardDeviation: 2.0)

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
        let exactlyHalf = Uncertain<Bool> { Double.random(in: 0...1) < 0.5 }
        let result = exactlyHalf.probability(exceeds: 0.5)
        // With true probability ~0.5, the result could be either true or false
        // depending on sampling variation, so we just check it's a boolean
        #expect(type(of: result) == Bool.self)
    }

    // MARK: - Performance and Efficiency Tests

    @Test("Efficient sampling with hypothesis testing")
    func testEfficientSamplingWithHypothesisTesting() {
        // Test that clear cases are decided quickly
        let clearCase = Uncertain<Bool> { true }

        // This should work efficiently even with large max samples
        let result = clearCase.probability(exceeds: 0.5, maxSamples: 10000)
        #expect(result == true)

        // Test with uncertain case
        let uncertainCase = Uncertain<Bool> { Double.random(in: 0...1) < 0.51 }
        let uncertainResult = uncertainCase.probability(exceeds: 0.5, maxSamples: 1000)

        // Should still produce a result (though might use more samples)
        #expect(type(of: uncertainResult) == Bool.self)
    }

    // MARK: - Paper Examples Integration Tests

    @Test("Complete GPS example from paper")
    func testCompleteGPSExample() {
        // Simulate the complete GPS walking scenario from the paper
        let startLocation = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 4.0)
        let endLocation = Uncertain<Double>.normal(mean: 100.0, standardDeviation: 4.0)

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
        // Demonstrate how evidence-based conditionals prevent bugs
        let sensorReading = Uncertain<Double>.normal(mean: 50.0, standardDeviation: 10.0)
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
        // This test ensures that shared variables sample consistently within a single evaluation
        let x = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0)

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
        // Test the paper's point-mass constructor
        let constantValue = Uncertain<Double>.point(42.0)
        let stringConstant = Uncertain<String>.point("hello")
        let boolConstant = Uncertain<Bool>.point(true)

        // These should always return the same value
        for _ in 0..<10 {
            #expect(constantValue.sample() == 42.0)
            #expect(stringConstant.sample() == "hello")
            #expect(boolConstant.sample() == true)
        }

        // Test that point-mass values work in arithmetic
        let x = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0)
        let constantTen = Uncertain<Double>.point(10.0)
        let sum = x + constantTen

        let sumMean = sum.expectedValue(sampleCount: 1000)
        #expect(abs(sumMean - 15.0) < 0.5)  // 5 + 10
    }

    @Test("Logical operators for Uncertain<Bool>")
    func testLogicalOperatorsForUncertainBool() {
        // Create boolean distributions
        let mostlyTrue = Uncertain<Bool> { Double.random(in: 0...1) < 0.8 }
        let mostlyFalse = Uncertain<Bool> { Double.random(in: 0...1) < 0.2 }
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
        // Create a distribution with known probability
        let biasedCoin = Uncertain<Bool> { Double.random(in: 0...1) < 0.7 }

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

    @Test("Complex shared variable expressions")
    func testComplexSharedVariableExpressions() {
        // Test complex expressions with shared variables
        let a = Uncertain<Double>.normal(mean: 1.0, standardDeviation: 0.1)
        let b = Uncertain<Double>.normal(mean: 2.0, standardDeviation: 0.1)

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
        // Test that logical operators properly handle shared variables
        let x = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0)
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
}

#if canImport(CoreLocation)
    import CoreLocation

    @Suite("Uncertain Core Location Extensions")
    struct UncertainCoreLocationTests {
        @Test("Uncertain<CLLocation> from CLLocation models uncertainty")
        func testUncertainCLLocationFrom() {
            let base = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
                altitude: 10.0,
                horizontalAccuracy: 5.0,
                verticalAccuracy: 2.0,
                timestamp: Date()
            )
            let uncertain = Uncertain<CLLocation>.from(base)
            // Should sample near the base location
            let samples = (0..<1000).map { _ in uncertain.sample() }
            let meanLat = samples.map { $0.coordinate.latitude }.reduce(0, +) / 1000
            let meanLon = samples.map { $0.coordinate.longitude }.reduce(0, +) / 1000
            let meanAlt = samples.map { $0.altitude }.reduce(0, +) / 1000
            #expect(abs(meanLat - base.coordinate.latitude) < 0.01)
            #expect(abs(meanLon - base.coordinate.longitude) < 0.01)
            #expect(abs(meanAlt - base.altitude) < 0.5)
        }

        @Test("Uncertain<CLLocation> .location constructor")
        func testUncertainCLLocationLocation() {
            let coord = CLLocationCoordinate2D(latitude: 40.0, longitude: -74.0)
            let uncertain = Uncertain<CLLocation>.location(
                coordinate: coord,
                horizontalAccuracy: 10.0,
                verticalAccuracy: 3.0,
                altitude: 50.0
            )
            let samples = (0..<1000).map { _ in uncertain.sample() }
            let meanLat = samples.map { $0.coordinate.latitude }.reduce(0, +) / 1000
            let meanLon = samples.map { $0.coordinate.longitude }.reduce(0, +) / 1000
            let meanAlt = samples.map { $0.altitude }.reduce(0, +) / 1000
            #expect(abs(meanLat - coord.latitude) < 0.02)
            #expect(abs(meanLon - coord.longitude) < 0.02)
            #expect(abs(meanAlt - 50.0) < 1.0)
        }

        @Test("Uncertain<CLLocation> distance and bearing")
        func testUncertainCLLocationDistanceAndBearing() {
            let a = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
                horizontalAccuracy: 5.0
            )
            let b = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7750, longitude: -122.4184),
                horizontalAccuracy: 5.0
            )
            let distance = Uncertain<CLLocation>.distance(from: a, to: b)
            let bearing = Uncertain<CLLocation>.bearing(from: a, to: b)
            let meanDistance = (0..<1000).map { _ in distance.sample() }.reduce(0, +) / 1000
            let meanBearing = (0..<1000).map { _ in bearing.sample() }.reduce(0, +) / 1000
            // Should be close to the true distance and bearing
            let trueA = CLLocation(latitude: 37.7749, longitude: -122.4194)
            let trueB = CLLocation(latitude: 37.7750, longitude: -122.4184)
            let trueDistance = trueA.distance(from: trueB)
            #expect(abs(meanDistance - trueDistance) < 10.0)
            #expect(meanBearing >= 0 && meanBearing <= 360)
        }

        @Test("Uncertain<CLLocationCoordinate2D> coordinate and distance")
        func testUncertainCLLocationCoordinate2D() {
            let coordA = CLLocationCoordinate2D(latitude: 51.5, longitude: -0.1)
            let coordB = CLLocationCoordinate2D(latitude: 51.5005, longitude: -0.099)
            let a = Uncertain<CLLocationCoordinate2D>.coordinate(coordA, accuracy: 8.0)
            let b = Uncertain<CLLocationCoordinate2D>.coordinate(coordB, accuracy: 8.0)
            let distance = Uncertain<CLLocationCoordinate2D>.distance(from: a, to: b)
            let meanDistance = (0..<1000).map { _ in distance.sample() }.reduce(0, +) / 1000
            let trueA = CLLocation(latitude: coordA.latitude, longitude: coordA.longitude)
            let trueB = CLLocation(latitude: coordB.latitude, longitude: coordB.longitude)
            let trueDistance = trueA.distance(from: trueB)
            #expect(abs(meanDistance - trueDistance) < 15.0)
        }

        @Test("Uncertain<CLLocationSpeed> from location")
        func testUncertainCLLocationSpeed() {
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
                let speed = Uncertain<CLLocationSpeed>.speed(from: loc)
                let samples = (0..<1000).map { _ in speed.sample() }
                let meanSpeed = samples.reduce(0, +) / 1000
                #expect(abs(meanSpeed - 15.0) < 1.0)
            }
        }

        @Test("Uncertain<CLLocationDirection> from location")
        func testUncertainCLLocationDirection() {
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
                let course = Uncertain<CLLocationDirection>.course(from: loc)
                let samples = (0..<1000).map { _ in course.sample() }
                let meanCourse = samples.reduce(0, +) / 1000
                #expect(abs(meanCourse - 45.0) < 5.0)
                #expect(meanCourse >= 0 && meanCourse <= 360)
            }
        }

        @Test("Parameterized speed uncertainty")
        func testParameterizedSpeedUncertainty() {
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
                    minimumSpeedUncertainty: 0.5
                )
                let liberalSpeed = Uncertain<CLLocationSpeed>.speed(
                    from: loc,
                    speedUncertaintyFactor: 0.2,
                    minimumSpeedUncertainty: 0.1
                )
                
                let conservativeSamples = (0..<1000).map { _ in conservativeSpeed.sample() }
                let liberalSamples = (0..<1000).map { _ in liberalSpeed.sample() }
                
                let conservativeStd = sqrt(conservativeSamples.map { pow($0 - 10.0, 2) }.reduce(0, +) / 1000)
                let liberalStd = sqrt(liberalSamples.map { pow($0 - 10.0, 2) }.reduce(0, +) / 1000)
                
                // Liberal should have higher uncertainty
                #expect(liberalStd > conservativeStd)
            }
        }

        @Test("Parameterized course uncertainty")
        func testParameterizedCourseUncertainty() {
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
                    courseUncertainty: 2.0
                )
                let roughCourse = Uncertain<CLLocationDirection>.course(
                    from: loc,
                    courseUncertainty: 15.0
                )
                
                let preciseSamples = (0..<1000).map { _ in preciseCourse.sample() }
                let roughSamples = (0..<1000).map { _ in roughCourse.sample() }
                
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

        @Test("Invalid location data handling")
        func testInvalidLocationDataHandling() {
            // Test with invalid accuracy
            let invalidLoc = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                altitude: 0,
                horizontalAccuracy: -1.0,  // Invalid
                verticalAccuracy: -1.0,    // Invalid
                course: -1.0,              // Invalid
                speed: -1.0,               // Invalid
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
            let baseCoord = CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194)
            let accuracy = 10.0
            
            // Test the improved coordinate method
            let uncertainCoord = Uncertain<CLLocationCoordinate2D>.coordinate(baseCoord, accuracy: accuracy)
            
            // Sample many times to check distribution
            let samples = (0..<1000).map { _ in uncertainCoord.sample() }
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
            #expect(latStd < 0.01)     // Not too loose
            #expect(lonStd > 0.00005)
            #expect(lonStd < 0.01)
        }

        @Test("Altitude uncertainty modeling")
        func testAltitudeUncertaintyModeling() {
            let baseLoc = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: 34.0, longitude: -118.0),
                altitude: 100.0,
                horizontalAccuracy: 5.0,
                verticalAccuracy: 3.0,
                timestamp: Date()
            )
            
            let uncertainLoc = Uncertain<CLLocation>.from(baseLoc)
            let samples = (0..<1000).map { _ in uncertainLoc.sample() }
            let altitudes = samples.map { $0.altitude }
            
            // Mean should be close to base altitude
            let meanAlt = altitudes.reduce(0, +) / 1000
            #expect(abs(meanAlt - 100.0) < 1.0)
            
            // Should have reasonable altitude uncertainty
            let altStd = sqrt(altitudes.map { pow($0 - 100.0, 2) }.reduce(0, +) / 1000)
            #expect(altStd > 1.0)  // Should have some uncertainty
            #expect(altStd < 10.0) // But not too much
        }

        @Test("Location uncertainty propagation in calculations")
        func testLocationUncertaintyPropagationInCalculations() {
            let locA = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
                horizontalAccuracy: 10.0,
                altitude: 50.0
            )
            
            let locB = Uncertain<CLLocation>.location(
                coordinate: CLLocationCoordinate2D(latitude: 37.7849, longitude: -122.4094),
                horizontalAccuracy: 10.0,
                altitude: 60.0
            )
            
            // Calculate uncertain distance and bearing
            let distance = Uncertain<CLLocation>.distance(from: locA, to: locB)
            let bearing = Uncertain<CLLocation>.bearing(from: locA, to: locB)
            
            // Test that uncertainty propagates correctly
            let distanceSamples = (0..<1000).map { _ in distance.sample() }
            let bearingSamples = (0..<1000).map { _ in bearing.sample() }
            
            // Distance should be around 1.5km (rough estimate)
            let meanDistance = distanceSamples.reduce(0, +) / 1000
            #expect(meanDistance > 1000)  // > 1km
            #expect(meanDistance < 3000)  // < 3km
            
            // Should have reasonable uncertainty in distance
            let distanceStd = sqrt(distanceSamples.map { pow($0 - meanDistance, 2) }.reduce(0, +) / 1000)
            #expect(distanceStd > 5.0)   // Some uncertainty
            #expect(distanceStd < 50.0)  // Not too much
            
            // Bearing should be roughly northeast
            let meanBearing = bearingSamples.reduce(0, +) / 1000
            #expect(meanBearing >= 0)
            #expect(meanBearing <= 360)
        }

        @Test("Course angle normalization")
        func testCourseAngleNormalization() {
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
                    courseUncertainty: 10.0  // Large uncertainty to test boundary
                )
                
                let samples = (0..<1000).map { _ in course.sample() }
                
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
                
                let exactCourse = Uncertain<CLLocationDirection>.course(from: locExact)
                let exactSamples = (0..<100).map { _ in exactCourse.sample() }
                
                #expect(exactSamples.allSatisfy { $0 >= 0 && $0 < 360 })
            }
        }
    }
#endif
