package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"TB2_concurrente/models/regression"
)

// loadDataset reads a CSV file and splits it into features (X) and targets (Y)
func loadDataset(path string) ([][]float64, []float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	headers, err := reader.Read()
	if err != nil {
		return nil, nil, err
	}

	var X [][]float64
	var Y []float64
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		var features []float64
		for i := 0; i < len(record)-1; i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, nil, err
			}
			features = append(features, val)
		}

		target, err := strconv.ParseFloat(record[len(record)-1], 64)
		if err != nil {
			return nil, nil, err
		}

		X = append(X, features)
		Y = append(Y, target)
	}

	fmt.Printf("Dataset loaded: %d samples, %d features\n", len(X), len(headers)-1)
	return X, Y, nil
}

// calculateMSE computes the Mean Squared Error
func calculateMSE(preds, actuals []float64) float64 {
	sum := 0.0
	for i := 0; i < len(preds); i++ {
		diff := preds[i] - actuals[i]
		sum += diff * diff
	}
	return sum / float64(len(preds))
}

// calculateAccuracy for logistic regression classification
func calculateAccuracy(preds, actuals []float64) float64 {
	correct := 0
	for i := 0; i < len(preds); i++ {
		if preds[i] == actuals[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(preds))
}

func main() {
	rand.Seed(time.Now().UnixNano())

	XTrain, YTrain, err := loadDataset("dataset/california_housing_train.csv")
	if err != nil {
		log.Fatalf("Error loading training data: %v", err)
	}

	XTest, YTest, err := loadDataset("dataset/california_housing_test.csv")
	if err != nil {
		log.Fatalf("Error loading test data: %v", err)
	}

	// ======= Sequential Linear Regression =======
	fmt.Println("\n[Sequential Linear Regression]")
	modelSeq := regression.NewLinearRegression(0.0001, 100)

	startSeq := time.Now()
	modelSeq.Train(XTrain, YTrain)
	elapsedSeq := time.Since(startSeq)

	predsSeq := modelSeq.Predict(XTest)
	mseSeq := calculateMSE(predsSeq, YTest)

	fmt.Printf("Training time: %s\n", elapsedSeq)
	fmt.Printf("First 3 weights: %.4f %.4f %.4f ...\n", modelSeq.Weights[0], modelSeq.Weights[1], modelSeq.Weights[2])
	fmt.Printf("Bias: %.4f\n", modelSeq.Bias)
	fmt.Printf("MSE: %.4f\n", mseSeq)

	// ======= Concurrent Linear Regression =======
	fmt.Println("\n[Concurrent Linear Regression]")
	modelCon := regression.NewLinearRegressionConcurrent(0.0001, 100)

	startCon := time.Now()
	modelCon.TrainConcurrently(XTrain, YTrain, 8)
	elapsedCon := time.Since(startCon)

	predsCon := modelCon.Predict(XTest)
	mseCon := calculateMSE(predsCon, YTest)

	fmt.Printf("Training time: %s\n", elapsedCon)
	fmt.Printf("First 3 weights: %.4f %.4f %.4f ...\n", modelCon.Weights[0], modelCon.Weights[1], modelCon.Weights[2])
	fmt.Printf("Bias: %.4f\n", modelCon.Bias)
	fmt.Printf("MSE: %.4f\n", mseCon)

	// ======= Logistic Regression (Sequential) =======
	fmt.Println("\n[Sequential Logistic Regression]")
	logXTrain, logYTrain, err := loadDataset("dataset/tabular-benchmark_train.csv")
	if err != nil {
		log.Fatalf("Error loading logistic train data: %v", err)
	}

	logXTest, logYTest, err := loadDataset("dataset/tabular-benchmark_test.csv")
	if err != nil {
		log.Fatalf("Error loading logistic test data: %v", err)
	}

	logModel := regression.NewLogisticRegression(0.01, 500)
	startLog := time.Now()
	logModel.Train(logXTrain, logYTrain)
	elapsedLog := time.Since(startLog)

	predsLog := logModel.Predict(logXTest)
	accLog := calculateAccuracy(predsLog, logYTest)

	fmt.Printf("Training time: %s\n", elapsedLog)
	fmt.Printf("First 3 weights: %.4f %.4f %.4f ...\n", logModel.Weights[0], logModel.Weights[1], logModel.Weights[2])
	fmt.Printf("Bias: %.4f\n", logModel.Bias)
	fmt.Printf("Accuracy: %.2f%%\n", accLog*100)

	// ======= Logistic Regression (Concurrent) =======
	fmt.Println("\n[Concurrent Logistic Regression]")
	logModelCon := regression.NewLogisticRegressionConcurrent(0.01, 500)
	startLogCon := time.Now()
	logModelCon.TrainConcurrently(logXTrain, logYTrain, 8)
	elapsedLogCon := time.Since(startLogCon)

	predsLogCon := logModelCon.Predict(logXTest)
	accLogCon := calculateAccuracy(predsLogCon, logYTest)

	fmt.Printf("Training time: %s\n", elapsedLogCon)
	fmt.Printf("First 3 weights: %.4f %.4f %.4f ...\n", logModelCon.Weights[0], logModelCon.Weights[1], logModelCon.Weights[2])
	fmt.Printf("Bias: %.4f\n", logModelCon.Bias)
	fmt.Printf("Accuracy: %.2f%%\n", accLogCon*100)
}
