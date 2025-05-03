package regression

// LogisticRegression is a basic logistic regression model
// used for binary classification tasks
type LogisticRegression struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Epochs       int
}

// NewLogisticRegression initializes a new logistic regression model
func NewLogisticRegression(learningRate float64, epochs int) *LogisticRegression {
	return &LogisticRegression{
		LearningRate: learningRate,
		Epochs:       epochs,
	}
}

// Train trains the logistic regression model using gradient descent
func (lr *LogisticRegression) Train(X [][]float64, Y []float64) {
	numSamples := len(X)
	numFeatures := len(X[0])
	lr.Weights = make([]float64, numFeatures)

	for epoch := 0; epoch < lr.Epochs; epoch++ {
		gradW := make([]float64, numFeatures)
		gradB := 0.0

		for i := 0; i < numSamples; i++ {
			z := lr.dotProduct(lr.Weights, X[i]) + lr.Bias
			yPred := sigmoid(z)
			error := yPred - Y[i]

			for j := 0; j < numFeatures; j++ {
				gradW[j] += error * X[i][j]
			}
			gradB += error
		}

		for j := 0; j < numFeatures; j++ {
			lr.Weights[j] -= lr.LearningRate * gradW[j] / float64(numSamples)
		}
		lr.Bias -= lr.LearningRate * gradB / float64(numSamples)
	}
}

// Predict returns predictions (0 or 1) for a batch of samples
func (lr *LogisticRegression) Predict(X [][]float64) []float64 {
	predictions := make([]float64, len(X))
	for i := range X {
		predictions[i] = lr.PredictOne(X[i])
	}
	return predictions
}

// PredictOne returns the binary prediction for a single sample
func (lr *LogisticRegression) PredictOne(X []float64) float64 {
	y := sigmoid(lr.dotProduct(lr.Weights, X) + lr.Bias)
	if y >= 0.5 {
		return 1.0
	}
	return 0.0
}

// dotProduct computes the dot product between two vectors
func (lr *LogisticRegression) dotProduct(v1, v2 []float64) float64 {
	sum := 0.0
	for i := range v1 {
		sum += v1[i] * v2[i]
	}
	return sum
}
