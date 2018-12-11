#!/bin/sh
# Compiles glsl to SPIR-V

cd shaders
glslangValidator -V shader.vert
glslangValidator -V shader.frag
