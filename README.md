# Uncertain\<T\>

A Swift library for uncertainty-aware programming,
which is especially useful for making reliable decisions with 
noisy, error-prone, or incomplete data.

This library implements the approach described in:

> **James Bornholt, Todd Mytkowicz, and Kathryn S. McKinley**  
> [*"Uncertain<T>: A First-Order Type for Uncertain Data"*][bornholt2014uncertain]  
> Architectural Support for Programming Languages and Operating Systems (ASPLOS), 
> March 2014.

💡 **Core Insight**:
Instead of *"Is this true?"*, 
ask *"How confident are we?"*

Traditional programs treat uncertain data as exact values,
leading to unreliable decisions:

```swift
// ❌ WRONG: Treats GPS reading as exact truth
let speed = gps.speed  // 55.2 mph (but GPS has ±5 mph error!)
if speed > 60.0 {
    issueSpeedingTicket()  // Oops! False positive due to GPS error
}
```

`Uncertain<T>` uses evidence-based conditionals that account for uncertainty:

```swift
import Uncertain

// ✅ CORRECT: Ask about evidence, not facts
let speed = Uncertain<Double>.gpsSpeed(reading: 55.2, accuracy: 5.0)
if (speed > 60.0).probability(exceeds: 0.95) {
    issueSpeedingTicket()  // Only ticket if 95% confident
}
```

## Requirements

- Swift 6.0+

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/mattt/Uncertain.git", from: "1.0.0")
]
```

## Usage

### Basic Usage

```swift
import Uncertain

// Create uncertain values
let temperature = Uncertain<Double>.normal(mean: 1.0, standardDeviation: 2.0)
let freezingPoint = 0.0

// Evidence-based conditional (the key insight!)
let freezingEvidence = temperature < freezingPoint

// Different confidence levels for different actions
if freezingEvidence.probability(exceeds: 0.95) {
    print("Turn on heater - 95% confident it's freezing")
} else if freezingEvidence.probability(exceeds: 0.3) {
    print("Monitor temperature - some chance of freezing")
} else {
    print("No action needed - unlikely to freeze")
}
```

### Evidence-Based Conditionals

The key innovation is **conditional semantics**. Instead of:

```swift
// Traditional approach - treats uncertain data as boolean facts
if uncertainValue > threshold {  // ❌ Uncertainty bug!
    takeAction()
}
```

Use evidence-based conditionals:

```swift
// Paper approach - asks about evidence
if (uncertainValue > threshold).probability(exceeds: 0.9) {  // ✅ Safe!
    takeAction()
}

// Or use implicit conditional (equivalent to probability(exceeds: 0.5))
if ~(uncertainValue > threshold) {  // ✅ Also safe!
    takeAction()
}
```

### Distribution Constructors

Use one of the buil-in probability distributions,
or create your own:

```swift
// Built-in distributions
let normal = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 1.0)
let uniform = Uncertain<Double>.uniform(min: 0.0, max: 1.0)
let exponential = Uncertain<Double>.exponential(rate: 1.0)

// Custom distributions
let custom = Uncertain<Double> {
    // Your sampling logic here
    return myRandomValue()
}
```

### Hypothesis Testing

Use Sequential Probability Ratio Test (SPRT) for efficient evidence evaluation:

```swift
let sensor = Uncertain<Double>.normal(mean: 25.0, standardDeviation: 3.0)
let evidence = sensor > 30.0

// Automatically determines sample size based on statistical significance
let highConfidence = evidence.probability(exceeds: 0.95, confidenceLevel: 0.99)
```

### Computation Graph

Operations build computation graphs, not immediate values,
so sampling happens only when evidence is evaluated.

```swift
let x = Uncertain<Double>.normal(mean: 10.0, standardDeviation: 2.0)
let y = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0)
let result = (x + y) * 2.0 - 3.0  // Lazy evaluation tree

let evidence = result > 15.0
let confident = evidence.probability(exceeds: 0.8)  // Now we sample
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

[bornholt2014uncertain]: https://www.microsoft.com/en-us/research/publication/uncertaint-a-first-order-type-for-uncertain-data-2/