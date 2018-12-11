VULKAN_SDK_PATH = /home/void/SDK/Vulkan/1.1.92.1/x86_64

CFLAGS = -std=c++17 -I$(VULKAN_SDK_PATH)/include

LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

helloTriangle: main.cpp
	@echo "Compiling Shaders: "
	@./compile_shaders.sh
	@echo "Compiling Executable: "
	g++ $(CFLAGS) -o helloTriangle main.cpp $(LDFLAGS)

.PHONY: test clean

test: helloTriangle
	 bash -c "source $(VULKAN_SDK_PATH)/../setup-env.sh && VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/explicit_layer.d ./helloTriangle "

clean:
	rm -f helloTriangle
