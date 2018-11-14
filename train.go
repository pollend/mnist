package main

import (
	"fmt"
	"github.com/pollend/GoMNIST"
	"os"
	"./Network"
)
func main() {
	// train, test
	train, _, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Println(fmt.Sprintf("failed to load mnist: (%s)", err))
		os.Exit(0)
	}

	net := Network.CreateNetwork([]int{16*16,20,10})
	sweeper := train.Sweep()
	for {
		//image label
		image, _, present := sweeper.Next()
		if !present {
			break
		}
		rect := image.Bounds()
		for x := rect.Min.X; x < rect.Max.X; x++{
			for y := rect.Min.Y; y < rect.Max.Y; x++{

			}
		}
		net.Predict()
	}

}
