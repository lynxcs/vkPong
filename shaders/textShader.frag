#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler[9];

layout(binding = 2) uniform sampleId {
    uint idX;
} score;

float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

void main() {
    float distanceFactor = 25;
    vec3 samples = texture(texSampler[score.idX], fragTexCoord).rgb;
    float sigDist = distanceFactor*(median(samples.r, samples.g, samples.b) - 0.5);
    float opacity = clamp(sigDist + 0.5, 0.0, 1.0);
    outColor = mix(vec4(0.0, 0.0, 0.0, 0.0), vec4(0.2, 0.2, 0.2, 1.0), opacity);
    if (outColor.a == 0.0)
        discard;
}

