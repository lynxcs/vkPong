#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 fragUV;

layout (location = 0) out vec4 outColor;

void main() {
    if (fragUV.x < 0.503 && fragUV.x > 0.497) {
        if (mod(fragUV.y, 0.07) < 0.03) {
            outColor = vec4(1.0, 1.0, 1.0, 1.0);
        } else {
            outColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
    } else {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
