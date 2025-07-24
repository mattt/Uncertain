# Uncertain\<T\>

A Swift library for uncertainty-aware programming,
which is especially useful for making reliable decisions with 
noisy, error-prone, or incomplete data.

This library implements the approach described in:

> **James Bornholt, Todd Mytkowicz, and Kathryn S. McKinley**  
> [*"Uncertain<T>: A First-Order Type for Uncertain Data"*][bornholt2014uncertain]  
> Architectural Support for Programming Languages and Operating Systems (ASPLOS), 
> March 2014.


Programs often treat uncertain data as exact values,
leading to unreliable decisions:

```swift
import CoreLocation

let speedLimit: CLLocationSpeed = 25 // 25 m/s ≈ 60 mph or 90 km/h

// ❌ WRONG: Treats GPS reading as exact truth
if location.speed > speedLimit { 
    issueSpeedingTicket()  // Oops! False positive due to GPS uncertainty
}
```

`Uncertain<T>` uses evidence-based conditionals that account for uncertainty:

```swift
import Uncertain
import CoreLocation

let speedLimit: CLLocationSpeed = 25 // 25 m/s ≈ 60 mph or 90 km/h

// ✅ CORRECT: Ask about evidence, not facts
let uncertainLocation = Uncertain<CLLocation>.from(location)
let uncertainSpeed = uncertainLocation.speed
if (uncertainSpeed > speedLimit).probability(exceeds: 0.95) {
    issueCitation()  // Only if 95% confident
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

```swift
import Uncertain

let temperatureSensor = Uncertain<Double>.normal(mean: 23.5, standardDeviation: 1.2)
let humiditySensor = Uncertain<Double>.normal(mean: 45.0, standardDeviation: 5.0)

// Combine sensor readings
let comfortIndex = temperatureSensor.map { temp in
    humiditySensor.map { humidity in
        // Heat index calculation
        -42.379 + 2.04901523 * temp + 10.14333127 * humidity
    }
}.flatMap { $0 }

let comfortable = (comfortIndex >= 68.0) && (comfortIndex <= 78.0)
if comfortable.probability(exceeds: 0.8) {
    print("Environment is comfortable")
}
```

For platforms that support the [Core Location framework][corelocation],
this library provides built-in convenience methods for working with GPS data:

```swift
import Uncertain
import CoreLocation

let userLocation = Uncertain<CLLocationCoordinate2D>.coordinate(
    CLLocationCoordinate2D(latitude: 37.7749, longitude: -122.4194),
    accuracy: 10.0  // ±10 meters
)

let destination = Uncertain<CLLocationCoordinate2D>.coordinate(
    CLLocationCoordinate2D(latitude: 37.7849, longitude: -122.4094),
    accuracy: 5.0 // ±5 meters
)

let distance = userLocation.distance(to: destination)
if (distance < 100.0).probability(exceeds: 0.9) {
    print("User is likely at the destination")
} else {
    let bearing = userLocation.bearing(to: destination)
    if (bearing >= 45.0 && bearing <= 135.0).probability(exceeds: 0.8) {
        print("User should head generally eastward")
    }
}
```

### Hypothesis Testing

Use [Sequential Probability Ratio Test][sprt] (SPRT) for efficient evidence evaluation:

```swift
let sensor = Uncertain<Double>.normal(mean: 25.0, standardDeviation: 3.0)
let evidence = sensor > 30.0

// Basic probability test
let isLikely = evidence.probability(exceeds: 0.95)

// Advanced hypothesis testing with full control
let result = evidence.evaluateHypothesis(
    threshold: 0.8,
    confidenceLevel: 0.99,
    maxSamples: 5000,
    epsilon: 0.05,
    alpha: 0.01,
    beta: 0.01,
    batchSize: 20
)

print("Decision: \(result.decision)")
print("Observed probability: \(result.probability)")
print("Samples used: \(result.samplesUsed)")
```

> [!TIP]
> The `alpha` and `beta` parameters correspond to the 
> [Type I error][type-i-error] (false positive rate) and 
> [Type II error][type-ii-error] (false negative rate), respectively. 
> Adjust these to control the strictness of your hypothesis test.

The SPRT approach automatically determines sample sizes based on statistical significance, 
making hypothesis testing both efficient and reliable.

### Operations

Operations build computation graphs rather than computing immediate values, 
so sampling only occurs when you evaluate evidence.

```swift
let x = Uncertain<Double>.normal(mean: 10.0, standardDeviation: 2.0)
let y = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0)

// Expressions are lazily evaluated
let result = (x + y) * 2.0 - 3.0 // Uncertain<Double>
let evidence = result > 15.0 // Uncertain<Bool>

// Now we evaluate
let confident = evidence.probability(exceeds: 0.8)  // Bool
```

#### Arithmetic Operations

All standard arithmetic operations are supported and build computation graphs:

```swift
let a = Uncertain<Double>.normal(mean: 10.0, standardDeviation: 2.0)
let b = Uncertain<Double>.normal(mean: 5.0, standardDeviation: 1.0)

// Arithmetic with other uncertain values
let sum = a + b
let difference = a - b
let product = a * b
let quotient = a / b

// Arithmetic with constants
let scaled = a * 2.0
let shifted = a + 5.0
let normalized = (a - 10.0) / 2.0
```

#### Comparison Operations

Comparisons return uncertain `Uncertain<Bool>`, not `Bool`:

```swift
let speed = Uncertain<Double>.normal(mean: 55.0, standardDeviation: 5.0)

// All comparisons return Uncertain<Bool>
let tooFast = speed > 60.0
let tooSlow = speed < 45.0
let exactMatch = speed == 55.0
let withinRange = (speed >= 50.0) && (speed <= 60.0)
```

#### Logical Operations

Boolean operations work with `Uncertain<Bool>` values:

```swift
let condition1 = speed > 50.0
let condition2 = speed < 70.0

let bothTrue = condition1 && condition2
let eitherTrue = condition1 || condition2
let notTrue = !condition1
```

#### Functional Operations

Transform uncertain values with `map` / `flatMap` / `filter`:

```swift
let temperature = Uncertain<Double>.normal(mean: 20.0, standardDeviation: 3.0)

// Map: transform each sample
let fahrenheit = temperature.map { $0 * 9/5 + 32 }

// FlatMap: chain uncertain computations
let adjusted = temperature.flatMap { temp in
    Uncertain<Double>.normal(mean: temp, standardDeviation: 1.0)
}

// Filter: [rejection sampling][rejection-sampling]
let positive = temperature.filter { $0 > 0 }
```

### Distribution Constructors

This library comes with built-in constructors for various
[probability distributions][probability-distribution].

> [!TIP]
> Check out [this companion project][Uncertain-Distribution-Visualizer],
> which has interactive visualizations to help you build up an intuition
> about these probability distributions.
>
> ![Screenshot of Uncertain Distribution Visualizer app](https://github.com/mattt/Uncertain-Distribution-Visualizer/raw/main/Assets/normal-dark.png)


#### [Normal][normal-distribution] ([Gaussian][normal-distribution]) Distribution

```swift
let normal = Uncertain<Double>.normal(mean: 0.0, standardDeviation: 1.0)
```

Best for modeling measurement errors, natural phenomena, and 
[central limit theorem][clt] applications.

#### [Uniform Distribution][uniform-distribution]

```swift
let uniform = Uncertain<Double>.uniform(min: 0.0, max: 1.0)
```

All values within the range are equally likely. 
Useful for random sampling and [Monte Carlo simulations][monte-carlo].

#### [Exponential Distribution][exponential-distribution]

```swift
let exponential = Uncertain<Double>.exponential(rate: 1.0)
```

Models time between events in a [Poisson process][poisson-process]. 
Common for modeling waiting times and lifetimes.

#### [Kumaraswamy Distribution][kumaraswamy-distribution]

```swift
let kumaraswamy = Uncertain<Double>.kumaraswamy(a: 2.0, b: 3.0)
```

A continuous distribution on [0,1] with flexible shapes. 
Similar to Beta distribution but with simpler mathematical forms.

#### [Rayleigh Distribution][rayleigh-distribution]

```swift
let rayleigh = Uncertain<Double>.rayleigh(scale: 1.0)
```

Models the magnitude of a 2D vector whose components are normally distributed.
Commonly used for modeling distances from a center point, such as GPS uncertainty.

#### [Binomial Distribution][binomial-distribution]

```swift
let binomial = Uncertain<Int>.binomial(trials: 100, probability: 0.3)
```

Models the number of successes in a fixed number of independent trials 
with constant success probability.

#### [Poisson Distribution][poisson-distribution]

```swift
let poisson = Uncertain<Int>.poisson(lambda: 3.5)
```

Models the number of events occurring in a fixed time interval 
when events occur independently at a constant rate.

#### [Bernoulli Distribution][bernoulli-distribution]

```swift
let bernoulli = Uncertain<Bool>.bernoulli(probability: 0.7)
```

Models a single trial with two possible outcomes (success/failure).

#### [Categorical Distribution][categorical-distribution]

```swift
let categorical = Uncertain<String>.categorical([
    "red": 0.3,
    "blue": 0.5,
    "green": 0.2
])
```

Models discrete outcomes with specified probabilities.

#### [Empirical Distribution][empirical-distribution]

```swift
let observedData = [1.2, 3.4, 2.1, 4.5, 1.8, 2.9]
let empirical = Uncertain<Double>.empirical(observedData)
```

Uses observed data to create a distribution 
by randomly sampling from the provided values.

#### [Mixture Distribution][mixture-distribution]

```swift
let mixture = Uncertain<Double>.mixture(
    of: [normal1, normal2, uniform],
    weights: [0.5, 0.3, 0.2]
)
```

Combines multiple distributions with optional weights.

#### Point Mass (Certain Value)

```swift
let certain = Uncertain<Double>.point(42.0)
```

Represents a known, certain value within the uncertain framework.

#### Custom Distributions

Or, you can always initialize with your own sampler:

```swift
let custom = Uncertain<Double> {
    // Your sampling logic here
    return myRandomValue()
}
```

The provided closure is called each time a sample is needed.

### Statistical Operations

`Uncertain<T>` provides convenience functions for statistical analysis:

#### [Expected Value][expected-value] (Mean)

```swift
let normal = Uncertain<Double>.normal(mean: 50.0, standardDeviation: 10.0)
let mean = normal.expectedValue(sampleCount: 1000)  // ≈ 50.0
```

#### [Standard Deviation][standard-deviation]

```swift
let std = normal.standardDeviation(sampleCount: 1000)  // ≈ 10.0
```

#### [Confidence Intervals][confidence-interval]

```swift
let (lower, upper) = normal.confidenceInterval(0.95, sampleCount: 1000)
// 95% of values fall between lower and upper bounds
```

#### [Skewness][skewness] and [Kurtosis][kurtosis]

```swift
let skew = normal.skewness(sampleCount: 1000)     // ≈ 0 for normal distribution
let kurt = normal.kurtosis(sampleCount: 1000)     // ≈ 0 for normal distribution
```

#### [Cumulative Distribution Function][cdf] (CDF)

```swift
let probability = normal.cdf(at: 60.0, sampleCount: 1000)
// Probability that a sample is ≤ 60.0
```

#### [Mode][mode] (Most Frequent Value)

```swift
let mode = categorical.mode(sampleCount: 1000)
// Most frequently occurring value
```

#### [Histogram][histogram]

```swift
let frequencies = categorical.histogram(sampleCount: 1000)
// Dictionary mapping values to occurrence counts
```

#### [Entropy][entropy]

```swift
let entropy = categorical.entropy(sampleCount: 1000)
// Information entropy in bits
```

#### [Log-Likelihood][log-likelihood] ([KDE][kernel-density-estimation])

```swift
let logLikelihood = normal.logLikelihood(45.0, sampleCount: 1000, bandwidth: 1.0)
// Estimated log-likelihood using kernel density estimation
```

## License

This project is available under the MIT license. 
See the LICENSE file for more info.

[bornholt2014uncertain]: https://www.microsoft.com/en-us/research/publication/uncertaint-a-first-order-type-for-uncertain-data-2/
[corelocation]: https://developer.apple.com/documentation/corelocation
[probability-distribution]: https://en.wikipedia.org/wiki/Probability_distribution
[normal-distribution]: https://en.wikipedia.org/wiki/Normal_distribution
[uniform-distribution]: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
[exponential-distribution]: https://en.wikipedia.org/wiki/Exponential_distribution
[binomial-distribution]: https://en.wikipedia.org/wiki/Binomial_distribution
[poisson-distribution]: https://en.wikipedia.org/wiki/Poisson_distribution
[bernoulli-distribution]: https://en.wikipedia.org/wiki/Bernoulli_distribution
[categorical-distribution]: https://en.wikipedia.org/wiki/Categorical_distribution
[empirical-distribution]: https://en.wikipedia.org/wiki/Empirical_distribution_function
[mixture-distribution]: https://en.wikipedia.org/wiki/Mixture_distribution
[clt]: https://en.wikipedia.org/wiki/Central_limit_theorem
[monte-carlo]: https://en.wikipedia.org/wiki/Monte_Carlo_method
[poisson-process]: https://en.wikipedia.org/wiki/Poisson_point_process
[sprt]: https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
[type-i-error]: https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#Type_I_error
[type-ii-error]: https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#Type_II_error
[expected-value]: https://en.wikipedia.org/wiki/Expected_value
[standard-deviation]: https://en.wikipedia.org/wiki/Standard_deviation
[confidence-interval]: https://en.wikipedia.org/wiki/Confidence_interval
[skewness]: https://en.wikipedia.org/wiki/Skewness
[kurtosis]: https://en.wikipedia.org/wiki/Kurtosis
[cdf]: https://en.wikipedia.org/wiki/Cumulative_distribution_function
[mode]: https://en.wikipedia.org/wiki/Mode_(statistics)
[histogram]: https://en.wikipedia.org/wiki/Histogram
[entropy]: https://en.wikipedia.org/wiki/Entropy_(information_theory)
[log-likelihood]: https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood
[kernel-density-estimation]: https://en.wikipedia.org/wiki/Kernel_density_estimation
[rejection-sampling]: https://en.wikipedia.org/wiki/Rejection_sampling
[kumaraswamy-distribution]: https://en.wikipedia.org/wiki/Kumaraswamy_distribution
[rayleigh-distribution]: https://en.wikipedia.org/wiki/Rayleigh_distribution
[Uncertain-Distribution-Visualizer]: https://github.com/mattt/Uncertain-Distribution-Visualizer/
