package regression

import (
	"sync"
)

// LinearRegressionConcurrent is a concurrent implementation of linear regression
type LinearRegressionConcurrent struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Epochs       int
}

// NewLinearRegressionConcurrent initializes a new concurrent regression model
func NewLinearRegressionConcurrent(learningRate float64, epochs int) *LinearRegressionConcurrent {
	return &LinearRegressionConcurrent{
		LearningRate: learningRate,
		Epochs:       epochs,
	}
}

// TrainConcurrently trains the model using multiple goroutines
func (lr *LinearRegressionConcurrent) TrainConcurrently(X [][]float64, Y []float64, numWorkers int) {
	numSamples := len(X)
	numFeatures := len(X[0])
	lr.Weights = make([]float64, numFeatures)

	for epoch := 0; epoch < lr.Epochs; epoch++ {
		gradW := make([]float64, numFeatures)
		gradB := 0.0

		var wg sync.WaitGroup
		mutex := &sync.Mutex{}

		chunkSize := (numSamples + numWorkers - 1) / numWorkers

		for w := 0; w < numWorkers; w++ {
			start := w * chunkSize
			end := start + chunkSize
			if end > numSamples {
				end = numSamples
			}

			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()

				localGradW := make([]float64, numFeatures)
				localGradB := 0.0

				for i := start; i < end; i++ {
					yPred := lr.PredictOne(X[i])
					err := yPred - Y[i]
					for j := 0; j < numFeatures; j++ {
						localGradW[j] += err * X[i][j]
					}
					localGradB += err
				}

				mutex.Lock()
				for j := 0; j < numFeatures; j++ {
					gradW[j] += localGradW[j]
				}
				gradB += localGradB
				mutex.Unlock()
			}(start, end)
		}

		wg.Wait()

		// Update weights and bias
		for j := 0; j < numFeatures; j++ {
			lr.Weights[j] -= lr.LearningRate * 2 * gradW[j] / float64(numSamples)
		}
		lr.Bias -= lr.LearningRate * 2 * gradB / float64(numSamples)
	}
}

// Predict returns the predictions for multiple samples
func (lr *LinearRegressionConcurrent) Predict(X [][]float64) []float64 {
	predictions := make([]float64, len(X))
	for i := range X {
		predictions[i] = lr.PredictOne(X[i])
	}
	return predictions
}

// PredictOne returns the prediction for a single sample
func (lr *LinearRegressionConcurrent) PredictOne(X []float64) float64 {
	return lr.dotProduct(lr.Weights, X) + lr.Bias
}

// dotProduct computes the dot product between two vectors
func (lr *LinearRegressionConcurrent) dotProduct(vec1, vec2 []float64) float64 {
	sum := 0.0
	for i := range vec1 {
		sum += vec1[i] * vec2[i]
	}
	return sum
}
