.PHONY: clean torch triton model_param

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm result/accuracy.txt

torch:
	python3 run_Torch_ResNet18.py

test-triton-relu:
	python3 run_TritonReLU_ResNet18.py

info:
	cat ResNet18/param_info.txt ResNet18/layer_info.txt

	