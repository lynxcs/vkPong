#!/bin/bash

VULKAN_SDK_PATH=/home/void/SDK/Vulkan/1.1.92.1/x86_64

source $VULKAN_SDK_PATH/../setup-env.sh
VK_LAYER_PATH=$VULKAN_SDK_PATH/etc/explicit_layer.d ./helloTriangle
