VULKAN_SDK_PATH = /home/void/SDK/Vulkan/1.1.92.1/x86_64

CFLAGS = -std=c++17 -I$(VULKAN_SDK_PATH)/include -g -Wall -Wextra -Wpedantic -Wno-unused-parameter

LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan -lshaderc_combined

vkPong: main.cpp
	clang++ $(CFLAGS) -o vkPong main.cpp $(LDFLAGS)

.PHONY: test clean

test: vkPong
	 bash -c "source $(VULKAN_SDK_PATH)/../setup-env.sh && VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/explicit_layer.d ./vkPong"

clean:
	rm -f vkPong
