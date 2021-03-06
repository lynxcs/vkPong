// TODO Use SPIRV-Reflect to fill out pipeline layout
// TODO Use dedicated transfer queue
// TODO Use dynamic uniform buffer
// TODO Use sampler2DArray instead of sampler2D[]

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/random.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <shaderc/shaderc.hpp>

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <thread>

#include <array>
#include <set>
#include <optional>
#include <fstream>

#include <chrono>
using namespace std::chrono_literals;

#ifdef NDEBUG
const bool VALIDATION_LAYERS_ENABLED = false;
#else
const bool VALIDATION_LAYERS_ENABLED = true;
#endif

const int MAX_FRAMES_IN_FLIGHT = 4;

const std::vector<const char*> c_validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> c_deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

float paddle1Pos = 0.0f;
float paddle2Pos = 0.0f;

bool keyWPressed = false;
bool keySPressed = false;
bool keyUpPressed = false;
bool keyDownPressed = false;

static void key_callback(GLFWwindow* f_window, int f_key, int f_scancode, int f_action, int f_mods) {
    if (f_key == GLFW_KEY_ESCAPE && f_action == GLFW_PRESS) {
        glfwSetWindowShouldClose(f_window, GLFW_TRUE);
    }

    if (f_key == GLFW_KEY_W && f_action == GLFW_PRESS) {
        keyWPressed = true;
    }

    if (f_key == GLFW_KEY_S && f_action == GLFW_PRESS) {
        keySPressed = true;
    }
    if (f_key == GLFW_KEY_UP && f_action == GLFW_PRESS) {
        keyUpPressed = true;
    }

    if (f_key == GLFW_KEY_DOWN && f_action == GLFW_PRESS) {
        keyDownPressed = true;
    }

    if (f_key == GLFW_KEY_W && f_action == GLFW_RELEASE) {
        keyWPressed = false;
    }

    if (f_key == GLFW_KEY_S && f_action == GLFW_RELEASE) {
        keySPressed = false;
    }
    if (f_key == GLFW_KEY_UP && f_action == GLFW_RELEASE) {
        keyUpPressed = false;
    }

    if (f_key == GLFW_KEY_DOWN && f_action == GLFW_RELEASE) {
        keyDownPressed = false;
    }
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    std::cerr << "VL: " << pCallbackData->pMessage << std::endl << std::endl;
    return VK_FALSE;
}

class application {
    public:
        void run() {
            initWindow();
            initVulkan();
            mainLoop();
            cleanup();
        }
    private:
        GLFWwindow* m_window;
        vk::Instance m_instance;
        vk::DebugUtilsMessengerEXT m_messenger;
        vk::SurfaceKHR m_surface;

        vk::PhysicalDevice m_physicalDevice;
        vk::Device m_device;

        vk::Queue m_graphicsQueue;
        vk::Queue m_presentQueue;

        vk::SwapchainKHR m_swapchain;
        vk::Format m_swapchainImageFormat;
        vk::Extent2D m_swapchainExtent;
        std::vector<vk::Image> m_swapchainImages;
        std::vector<vk::ImageView> m_swapchainImageViews;
        std::vector<vk::Framebuffer> m_swapchainFramebuffers;

        vk::RenderPass m_renderPass;

        vk::DescriptorSetLayout m_rect1DescriptorSetLayout;
        vk::DescriptorSetLayout m_rect2DescriptorSetLayout;
        vk::DescriptorSetLayout m_circleDescriptorSetLayout;
        vk::DescriptorSetLayout m_font1DescriptorSetLayout;
        vk::DescriptorSetLayout m_font2DescriptorSetLayout;

        vk::DescriptorPool m_objectDescriptorPool;
        vk::DescriptorPool m_textDescriptorPool;
        std::vector<vk::DescriptorSet> rect1DescriptorSets;
        std::vector<vk::DescriptorSet> rect2DescriptorSets;
        std::vector<vk::DescriptorSet> circleDescriptorSets;
        std::vector<vk::DescriptorSet> font1DescriptorSets;
        std::vector<vk::DescriptorSet> font2DescriptorSets;

        vk::PipelineLayout m_pipelineLayout;
        vk::Pipeline m_graphicsPipeline;

        vk::PipelineLayout m_backgroundPipelineLayout;
        vk::Pipeline m_backgroundPipeline;

        vk::PipelineLayout m_fontPipelineLayout;
        vk::Pipeline m_fontPipeline;

        vk::CommandPool m_commandPool;
        std::vector<vk::CommandBuffer> m_commandBuffers;

        std::vector<vk::Semaphore> m_imageAvailableSemaphores;
        std::vector<vk::Semaphore> m_renderFinishedSemaphores;
        std::vector<vk::Fence> m_inFlightFences;

        vk::Buffer m_rectVertexBuffer;
        vk::DeviceMemory m_rectVertexBufferMemory;

        vk::Buffer m_rectIndexBuffer;
        vk::DeviceMemory m_rectIndexBufferMemory;

        vk::Buffer m_circleVertexBuffer;
        vk::DeviceMemory m_circleVertexBufferMemory;

        vk::Buffer m_circleIndexBuffer;
        vk::DeviceMemory m_circleIndexBufferMemory;

        vk::Buffer m_fontVertexBuffer;
        vk::DeviceMemory m_fontVertexBufferMemory;

        vk::Buffer m_fontIndexBuffer;
        vk::DeviceMemory m_fontIndexBufferMemory;

        std::vector<vk::Buffer> m_circleUniformBuffers;
        std::vector<vk::DeviceMemory> m_circleUniformBuffersMemory;

        std::vector<vk::Buffer> m_rect1UniformBuffers;
        std::vector<vk::DeviceMemory> m_rect1UniformBuffersMemory;

        std::vector<vk::Buffer> m_rect2UniformBuffers;
        std::vector<vk::DeviceMemory> m_rect2UniformBuffersMemory;

        std::vector<vk::Buffer> m_text1IDUniformBuffers;
        std::vector<vk::DeviceMemory> m_text1IDUniformBuffersMemory;
        std::vector<vk::Buffer> m_text1MVPUniformBuffers;
        std::vector<vk::DeviceMemory> m_text1MVPUniformBuffersMemory;

        std::vector<vk::Buffer> m_text2IDUniformBuffers;
        std::vector<vk::DeviceMemory> m_text2IDUniformBuffersMemory;
        std::vector<vk::Buffer> m_text2MVPUniformBuffers;
        std::vector<vk::DeviceMemory> m_text2MVPUniformBuffersMemory;

        vk::Image m_fontImages[10];
        vk::DeviceMemory m_fontImagesMemory[10];
        vk::ImageView m_fontImageViews[10];
        vk::Sampler m_fontSamplers[10];

        int m_currentFrame = 0;

        bool m_framebufferResized = false;

        struct QueueFamilyIndices {
            std::optional<uint32_t> graphicsFamily;
            std::optional<uint32_t> presentFamily;

            bool isComplete() {
                return graphicsFamily.has_value() && presentFamily.has_value();
            }
        };

        struct UniformBufferObject {
            glm::mat4 mvpMatrix;
        };

        struct UniformScoreObject {
            glm::uint score;
        };

        struct MVPMatrixObject {
            glm::mat4 model;
            glm::mat4 view;
            glm::mat4 proj;
        };

        struct SwapchainSupportDetails {
            vk::SurfaceCapabilitiesKHR capabilities;
            std::vector<vk::SurfaceFormatKHR> formats;
            std::vector<vk::PresentModeKHR> presentModes;
        };

        struct Vertex {
            glm::vec2 pos = {0.0f, 0.0f};
            glm::vec3 color = {1.0f, 1.0f, 1.0f};

            static vk::VertexInputBindingDescription getBindingDescription() {
                vk::VertexInputBindingDescription bindingDescription = {};
                bindingDescription.binding = 0;
                bindingDescription.stride = sizeof(Vertex);
                bindingDescription.inputRate = vk::VertexInputRate::eVertex;

                return bindingDescription;
            }

            static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescription() {
                std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {};
                attributeDescriptions[0].binding = 0;
                attributeDescriptions[0].location = 0;
                attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
                attributeDescriptions[0].offset = offsetof(Vertex, pos);

                attributeDescriptions[1].binding = 0;
                attributeDescriptions[1].location = 1;
                attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
                attributeDescriptions[1].offset = offsetof(Vertex, color);

                return attributeDescriptions;
            }
        };

        struct fontVertex {
            glm::vec2 pos = {0.0f, 0.0f};
            glm::vec3 color = {1.0f, 1.0f, 1.0f};
            glm::vec2 texCoord;

            static vk::VertexInputBindingDescription getBindingDescription() {
                vk::VertexInputBindingDescription bindingDescription = {};
                bindingDescription.binding = 0;
                bindingDescription.stride = sizeof(fontVertex);
                bindingDescription.inputRate = vk::VertexInputRate::eVertex;

                return bindingDescription;
            }

            static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescription() {
                std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions = {};
                attributeDescriptions[0].binding = 0;
                attributeDescriptions[0].location = 0;
                attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
                attributeDescriptions[0].offset = offsetof(fontVertex, pos);

                attributeDescriptions[1].binding = 0;
                attributeDescriptions[1].location = 1;
                attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
                attributeDescriptions[1].offset = offsetof(fontVertex, color);

                attributeDescriptions[2].binding = 0;
                attributeDescriptions[2].location = 2;
                attributeDescriptions[2].format = vk::Format::eR32G32Sfloat;
                attributeDescriptions[2].offset = offsetof(fontVertex, texCoord);

                return attributeDescriptions;
            }
        };

        const std::vector<fontVertex> c_fontVertices {
            {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0, 0.0f}},
            {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0, 1.0f}},
            {{-0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}
        };

        const std::vector<uint16_t> c_fontIndices {
            0, 1, 2, 2, 3, 0
        };

        const std::vector<Vertex> c_rectVertices = {
            {{-0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}},
            {{0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}},
            {{0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
            {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
        };

        const std::vector<uint16_t> c_rectIndices = {
            0, 1, 2, 2, 3, 0
        };

        std::vector<Vertex> c_circleVertices;
        std::vector<uint16_t> c_circleIndices;

        static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
            auto app = reinterpret_cast<application*>(glfwGetWindowUserPointer(window));
            app->m_framebufferResized = true;
        }

        void initVulkan() {
            createInstance();
            setupDebugCallback();
            createSurface();
            pickPhysicalDevice();
            createLogicalDevice();
            createSwapchain();
            createImageViews();
            createRenderPass();
            createDescriptorSetLayout();

            createBackgroundPipeline();
            createObjectPipeline();
            createTextPipeline();

            createFramebuffers();
            createCommandPool();

            createFontStuff();
            createFontVertexBuffer();
            createFontIndexBuffer();

            createCircleMesh();
            createCircleVertexBuffer();
            createCircleIndexBuffer();

            createRectVertexBuffer();
            createRectIndexBuffer();

            createUniformBuffers();
            createDescriptorPool();
            createDescriptorSets();
            createCommandBuffers();
            createSyncObjects();
        }

        void createFontVertexBuffer() {
            using MP_FB = vk::MemoryPropertyFlagBits;
            using BU_FB = vk::BufferUsageFlagBits;

            vk::DeviceSize bufferSize = sizeof(c_fontVertices[0]) * c_fontVertices.size();

            vk::Buffer stagingBuffer;
            vk::DeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, BU_FB::eTransferSrc, MP_FB::eHostVisible | MP_FB::eHostCoherent, stagingBuffer, stagingBufferMemory);

            void* data;
            m_device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags(0), &data);
            memcpy(data, c_fontVertices.data(), (size_t) bufferSize);
            m_device.unmapMemory(stagingBufferMemory);

            createBuffer(bufferSize, BU_FB::eTransferDst | BU_FB::eVertexBuffer, MP_FB::eDeviceLocal, m_fontVertexBuffer, m_fontVertexBufferMemory);

            vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
            copyBuffer(commandBuffer, stagingBuffer, m_fontVertexBuffer, bufferSize);
            flushSingleTimeCommands(commandBuffer);

            m_device.destroy(stagingBuffer);
            m_device.freeMemory(stagingBufferMemory);
        }

        void createFontIndexBuffer() {
            using MP_FB = vk::MemoryPropertyFlagBits;
            using BU_FB = vk::BufferUsageFlagBits;

            vk::DeviceSize bufferSize = sizeof(c_fontIndices[0]) * c_fontIndices.size();

            vk::Buffer stagingBuffer;
            vk::DeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, BU_FB::eTransferSrc, MP_FB::eHostVisible | MP_FB::eHostCoherent, stagingBuffer, stagingBufferMemory);

            void* data;
            m_device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags(0), &data);
            memcpy(data, c_fontIndices.data(), (size_t) bufferSize);
            m_device.unmapMemory(stagingBufferMemory);

            createBuffer(bufferSize, BU_FB::eTransferDst | BU_FB::eIndexBuffer, MP_FB::eDeviceLocal, m_fontIndexBuffer, m_fontIndexBufferMemory);

            vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
            copyBuffer(commandBuffer, stagingBuffer, m_fontIndexBuffer, bufferSize);
            flushSingleTimeCommands(commandBuffer);

            m_device.destroy(stagingBuffer);
            m_device.freeMemory(stagingBufferMemory);
        }

        void createFontStuff() {
            for(int i = 0; i < 10; i++) {
                std::string filename = "textures/font_" + std::to_string(i) + ".png";
                int texWidth, texHeight, texChannels;
                stbi_uc* pixels = stbi_load(filename.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
                vk::DeviceSize imageSize = texWidth * texHeight * 4;

                if (!pixels)
                    throw std::runtime_error("Failed to load texture image!");

                vk::Buffer stagingBuffer;
                vk::DeviceMemory stagingBufferMemory;

                using BU = vk::BufferUsageFlagBits;
                using MP = vk::MemoryPropertyFlagBits;
                createBuffer(imageSize, BU::eTransferSrc, MP::eHostVisible | MP::eHostCoherent, stagingBuffer, stagingBufferMemory);

                void* data;
                m_device.mapMemory(stagingBufferMemory,0, imageSize, vk::MemoryMapFlags(0), &data);
                memcpy(data, pixels, static_cast<size_t>(imageSize));
                m_device.unmapMemory(stagingBufferMemory);

                stbi_image_free(pixels);

                vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
                createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, m_fontImages[i], m_fontImagesMemory[i]);
                transitionImageLayout(commandBuffer, m_fontImages[i], vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
                copyBufferToImage(commandBuffer, stagingBuffer, m_fontImages[i], static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
                transitionImageLayout(commandBuffer, m_fontImages[i], vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
                flushSingleTimeCommands(commandBuffer);

                m_device.destroy(stagingBuffer);
                m_device.free(stagingBufferMemory);

                m_fontImageViews[i] = createImageView(m_fontImages[i], vk::Format::eR8G8B8A8Unorm);

                vk::SamplerCreateInfo samplerInfo = {};
                samplerInfo.magFilter = vk::Filter::eLinear;
                samplerInfo.minFilter = vk::Filter::eLinear;

                samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
                samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
                samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;

                samplerInfo.anisotropyEnable = VK_TRUE;
                samplerInfo.maxAnisotropy = 16;
                samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
                samplerInfo.unnormalizedCoordinates = VK_FALSE;

                samplerInfo.compareEnable = VK_FALSE;
                samplerInfo.compareOp = vk::CompareOp::eAlways;

                samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
                samplerInfo.mipLodBias = 0.0f;
                samplerInfo.minLod = 0.0f;
                samplerInfo.maxLod = 0.0f;

                m_device.createSampler(&samplerInfo, nullptr, &m_fontSamplers[i]);
            }
        }

        void createCircleMesh() {
            std::vector<Vertex> circleVerts;
            std::vector<unsigned short> circleIndices;
            circleVerts.push_back(Vertex{ glm::vec3(0, 0, 0) });

            for (int i = 0; i < 360; ++i)
            {
                glm::vec3 vert;
                //   |   /|
                //   |  / |
                //   |-/ <|-- angle i
                //   |/___|
                //  opposite = s=o/h sin(i) * hyp = o;
                //  adj = c=a/h cos(i) * hyp = a;
                //	|hyp| == 1

                vert.x = sinf((float)i);
                vert.y = cosf((float)i);
                vert.z = 0;
                circleVerts.push_back(Vertex{ vert });

                if (i > 0)
                {
                    circleIndices.push_back(0);
                    circleIndices.push_back(i + 1);
                    circleIndices.push_back(i);
                }
            }

            c_circleVertices = circleVerts;
            c_circleIndices = circleIndices;
        }

        void createInstance() {

            std::vector<vk::ExtensionProperties> availableExtensions;
            availableExtensions = vk::enumerateInstanceExtensionProperties();

            std::vector<const char*> requiredExtensions;
            requiredExtensions = getRequiredExtensions();

            if (VALIDATION_LAYERS_ENABLED) {
                std::vector<vk::LayerProperties> availableLayers;
                availableLayers = vk::enumerateInstanceLayerProperties();

                std::string missingLayers = "";
                for (const auto& layerName : c_validationLayers) {
                    auto pred = [&](vk::LayerProperties x) { return !strcmp(layerName, x.layerName); };
                    if ( std::none_of(availableLayers.begin(), availableLayers.end(), pred)) {
                        missingLayers+= "\t";
                        missingLayers+= layerName;
                        missingLayers+= "\n";
                    }
                }

                if (missingLayers != "")
                    throw std::runtime_error("Missing requested layers: \n" + missingLayers);
            }

            if (requiredExtensions.size() == 0 || (requiredExtensions.size() == 1 && VALIDATION_LAYERS_ENABLED))
                throw std::runtime_error("Failed to get extensions required by GLFW!");

            std::string missingExtensions = "";
            for (const auto& requiredExtension : requiredExtensions) {
                auto pred = [&](vk::ExtensionProperties x) { return !strcmp(requiredExtension, x.extensionName); };
                if ( std::none_of(availableExtensions.begin(), availableExtensions.end(), pred)) {
                    missingExtensions+= "\t";
                    missingExtensions+= requiredExtension;
                    missingExtensions+= "\n";
                }
            }

            if (missingExtensions != "")
                throw std::runtime_error("Missing extensions required by GLFW: \n" + missingExtensions);

            vk::ApplicationInfo appInfo = {};
            appInfo.pApplicationName = "Hello Triangle";
            appInfo.applicationVersion = 1;
            appInfo.pEngineName = "No Engine";
            appInfo.engineVersion = 1;
            appInfo.apiVersion = VK_API_VERSION_1_1;

            vk::InstanceCreateInfo createInfo = {};
            createInfo.pApplicationInfo = &appInfo;

            createInfo.enabledExtensionCount = requiredExtensions.size();
            createInfo.ppEnabledExtensionNames = requiredExtensions.data();

            if (VALIDATION_LAYERS_ENABLED) {
                createInfo.enabledLayerCount = c_validationLayers.size();
                createInfo.ppEnabledLayerNames = c_validationLayers.data();
            } else {
                createInfo.enabledLayerCount = 0;
            }

            m_instance = vk::createInstance(createInfo);
        }

        void setupDebugCallback() {
            if (!VALIDATION_LAYERS_ENABLED)
                return;

            vk::DebugUtilsMessengerCreateInfoEXT createInfo = {};

            createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;

            createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral    |
                                     vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                                     vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

            createInfo.pfnUserCallback = debugCallback;
            createInfo.pUserData = nullptr;

            m_messenger = m_instance.createDebugUtilsMessengerEXT(createInfo, nullptr, vk::DispatchLoaderDynamic{m_instance});
        }

        void createSurface() {
            auto v_surface = static_cast<VkSurfaceKHR>(m_surface);
            if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &v_surface) != VK_SUCCESS)
                throw std::runtime_error("Failed to create window surface!");
            m_surface = v_surface;
        }

        void pickPhysicalDevice() {
            std::vector<vk::PhysicalDevice> v_devices = m_instance.enumeratePhysicalDevices();

            if (!v_devices.size())
                throw std::runtime_error("Failed to find GPUs with Vulkan support!");

            for (const auto& device : v_devices) {
                if (isDeviceSuitable(device)) {
                    m_physicalDevice = device;
                    break;
                }
            }

            if (!m_physicalDevice)
                throw std::runtime_error("Failed to find suitable GPU!");
        }

        void createLogicalDevice() {
            QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

            std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
            std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

            float queuePriority = 1.0f;
            for (uint32_t queueFamily : uniqueQueueFamilies) {
                vk::DeviceQueueCreateInfo queueCreateInfo = {};
                queueCreateInfo.queueFamilyIndex = queueFamily;
                queueCreateInfo.queueCount = 1;
                queueCreateInfo.pQueuePriorities = &queuePriority;
                queueCreateInfos.push_back(queueCreateInfo);
            }

            vk::PhysicalDeviceFeatures deviceFeatures = {};
            deviceFeatures.samplerAnisotropy = VK_TRUE;
            deviceFeatures.shaderSampledImageArrayDynamicIndexing = VK_TRUE;

            vk::DeviceCreateInfo createInfo = {};

            createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
            createInfo.pQueueCreateInfos = queueCreateInfos.data();

            createInfo.pEnabledFeatures = &deviceFeatures;

            createInfo.enabledExtensionCount = c_deviceExtensions.size();
            createInfo.ppEnabledExtensionNames = c_deviceExtensions.data();

            if (VALIDATION_LAYERS_ENABLED) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(c_validationLayers.size());
                createInfo.ppEnabledLayerNames = c_validationLayers.data();
            } else {
                createInfo.enabledLayerCount = 0;
            }

            m_device = m_physicalDevice.createDevice(createInfo, nullptr);

            m_graphicsQueue = m_device.getQueue(indices.graphicsFamily.value(), 0);
            m_presentQueue = m_device.getQueue(indices.presentFamily.value(), 0);
        }

        void createSwapchain() {
            SwapchainSupportDetails swapchainSupport = querySwapchainSupport(m_physicalDevice);
            vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapchainSupport.formats);
            vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapchainSupport.presentModes);
            vk::Extent2D extent = chooseSwapExtent(swapchainSupport.capabilities);

            uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
            if (imageCount > swapchainSupport.capabilities.maxImageCount && swapchainSupport.capabilities.maxImageCount > 0)
                imageCount = swapchainSupport.capabilities.maxImageCount;

            vk::SwapchainCreateInfoKHR createInfo = {};
            createInfo.surface = m_surface;
            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = surfaceFormat.format;
            createInfo.imageColorSpace = surfaceFormat.colorSpace;
            createInfo.imageExtent = extent;
            createInfo.imageArrayLayers = 1;
            createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

            QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);
            uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
            
            if (indices.graphicsFamily != indices.presentFamily) {
                createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
                createInfo.queueFamilyIndexCount = 2;
                createInfo.pQueueFamilyIndices = queueFamilyIndices;
            } else {
                createInfo.imageSharingMode = vk::SharingMode::eExclusive;
                
                // Optional
                createInfo.queueFamilyIndexCount = 2;
                createInfo.pQueueFamilyIndices = nullptr;
            }

            createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
            createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

            createInfo.presentMode = presentMode;
            createInfo.clipped = true;

            createInfo.oldSwapchain = nullptr;

            m_swapchain = m_device.createSwapchainKHR(createInfo, nullptr);

            m_swapchainImages = m_device.getSwapchainImagesKHR(m_swapchain);

            m_swapchainImageFormat = surfaceFormat.format;
            m_swapchainExtent = extent;
        }

        void createImageViews() {
            m_swapchainImageViews.resize(m_swapchainImages.size());
            for (uint32_t i = 0; i < m_swapchainImages.size(); i++) {
                m_swapchainImageViews[i] = createImageView(m_swapchainImages[i], m_swapchainImageFormat);
            }
        }

        void createRenderPass() {
            vk::AttachmentDescription colorAttachment = {};
            colorAttachment.format = m_swapchainImageFormat;
            colorAttachment.samples = vk::SampleCountFlagBits::e1;

            colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
            colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;

            colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;

            colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
            colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

            vk::AttachmentReference colorAttachmentRef = {};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

            vk::SubpassDescription subpass = {};
            subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;

            vk::SubpassDependency dependency = {};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;

            dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            dependency.srcAccessMask = vk::AccessFlags(0);

            dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;

            vk::RenderPassCreateInfo createInfo = {};
            createInfo.attachmentCount = 1;
            createInfo.pAttachments = &colorAttachment;
            createInfo.subpassCount = 1;
            createInfo.pSubpasses = &subpass;
            createInfo.dependencyCount = 1;
            createInfo.pDependencies = &dependency;

            m_renderPass = m_device.createRenderPass(createInfo, nullptr);
        }

        void createDescriptorSetLayout() {

            // Object descriptor set layout
            {
                vk::DescriptorSetLayoutBinding uboLayoutBinding = {};
                uboLayoutBinding.binding = 0;
                uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
                uboLayoutBinding.descriptorCount = 1;

                uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
                uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

                std::array<vk::DescriptorSetLayoutBinding, 1> objectBindings = {uboLayoutBinding};
                vk::DescriptorSetLayoutCreateInfo objectLayoutInfo = {};
                objectLayoutInfo.bindingCount = static_cast<uint32_t>(objectBindings.size());
                objectLayoutInfo.pBindings = objectBindings.data();

                m_rect1DescriptorSetLayout = m_device.createDescriptorSetLayout(objectLayoutInfo);
                m_rect2DescriptorSetLayout = m_device.createDescriptorSetLayout(objectLayoutInfo);
                m_circleDescriptorSetLayout = m_device.createDescriptorSetLayout(objectLayoutInfo);
            }

            // Font descriptor set layout
            {
                // Text MVP matrix
                vk::DescriptorSetLayoutBinding uboMVPLayoutBinding = {};
                uboMVPLayoutBinding.binding = 0;
                uboMVPLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
                uboMVPLayoutBinding.descriptorCount = 1;
                uboMVPLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
                uboMVPLayoutBinding.pImmutableSamplers = nullptr; // Optional

                // Array of sampler2D's
                vk::DescriptorSetLayoutBinding samplerLayoutBinding = {};
                samplerLayoutBinding.binding = 1;
                samplerLayoutBinding.descriptorCount = 10;
                samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
                samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
                samplerLayoutBinding.pImmutableSamplers = nullptr;

                // ID value of sampler2D array
                vk::DescriptorSetLayoutBinding uboIDLayoutBinding = {};
                uboIDLayoutBinding.binding = 2;
                uboIDLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
                uboIDLayoutBinding.descriptorCount = 1;
                uboIDLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
                uboIDLayoutBinding.pImmutableSamplers = nullptr; // Optional

                std::array<vk::DescriptorSetLayoutBinding, 3> fontBindings = {uboMVPLayoutBinding, samplerLayoutBinding, uboIDLayoutBinding};

                vk::DescriptorSetLayoutCreateInfo fontLayoutInfo = {};
                fontLayoutInfo.bindingCount = static_cast<uint32_t>(fontBindings.size());
                fontLayoutInfo.pBindings = fontBindings.data();
                m_font1DescriptorSetLayout = m_device.createDescriptorSetLayout(fontLayoutInfo);
                m_font2DescriptorSetLayout = m_device.createDescriptorSetLayout(fontLayoutInfo);
            }
        }

        void createBackgroundPipeline() {
            auto vertFile = compileShaderToSpirv("shaders/backgroundShader.vert");
            auto fragFile = compileShaderToSpirv("shaders/backgroundShader.frag");

            auto vertShaderModule = createShaderModule(vertFile);
            auto fragShaderModule = createShaderModule(fragFile);

            vk::PipelineShaderStageCreateInfo vertShaderStageInfo = {};
            vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
            vertShaderStageInfo.module = vertShaderModule;
            vertShaderStageInfo.pName = "main";

            vk::PipelineShaderStageCreateInfo fragShaderStageInfo = {};
            fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";

            vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

            vk::PipelineVertexInputStateCreateInfo emptyInputState = {};
            emptyInputState.vertexAttributeDescriptionCount = 0;
            emptyInputState.pVertexAttributeDescriptions = nullptr;
            emptyInputState.vertexBindingDescriptionCount = 0;
            emptyInputState.pVertexBindingDescriptions = nullptr;

            vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {};
            inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            vk::Viewport viewport = {0.0, 0.0, (float) m_swapchainExtent.width, (float) m_swapchainExtent.height, 0.0, 1.0};
            vk::Rect2D scissor = {{0, 0}, m_swapchainExtent};

            vk::PipelineViewportStateCreateInfo viewportState = {};
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;

            vk::PipelineRasterizationStateCreateInfo rasterizer = {};
            rasterizer.depthClampEnable = VK_FALSE;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode = vk::PolygonMode::eFill;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = vk::CullModeFlagBits::eFront;
            rasterizer.frontFace = vk::FrontFace::eCounterClockwise;

            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f;
            rasterizer.depthBiasClamp = 0.0f;

            vk::PipelineMultisampleStateCreateInfo multisampling = {};
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
            multisampling.minSampleShading = 1.0f;
            multisampling.pSampleMask = nullptr;
            multisampling.alphaToCoverageEnable = VK_FALSE;
            multisampling.alphaToOneEnable = VK_FALSE;

            vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
            using CCF = vk::ColorComponentFlagBits;
            colorBlendAttachment.colorWriteMask = CCF::eR | CCF::eG | CCF::eB | CCF::eA;
            colorBlendAttachment.blendEnable = VK_FALSE;

            vk::PipelineColorBlendStateCreateInfo colorBlending = {};
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.logicOp = vk::LogicOp::eCopy;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;

            vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.setLayoutCount = 0;
            pipelineLayoutInfo.pSetLayouts = nullptr;

            if (m_device.createPipelineLayout(&pipelineLayoutInfo, nullptr, &m_backgroundPipelineLayout) != vk::Result::eSuccess)
                throw std::runtime_error("Failed to create pipeline layout!");

            vk::GraphicsPipelineCreateInfo pipelineInfo = {};
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;

            pipelineInfo.pVertexInputState = &emptyInputState;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = nullptr;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = nullptr;

            pipelineInfo.layout = m_backgroundPipelineLayout;
            pipelineInfo.renderPass = m_renderPass;
            pipelineInfo.subpass = 0;

            m_backgroundPipeline = m_device.createGraphicsPipeline(nullptr, pipelineInfo);
            if (!m_backgroundPipeline)
                throw std::runtime_error("Failed to create pipeline!");

            m_device.destroyShaderModule(fragShaderModule, nullptr);
            m_device.destroyShaderModule(vertShaderModule, nullptr);
        }

        void createTextPipeline() {
            auto vertFile = compileShaderToSpirv("shaders/textShader.vert");
            auto fragFile = compileShaderToSpirv("shaders/textShader.frag");

            auto vertShaderModule = createShaderModule(vertFile);
            auto fragShaderModule = createShaderModule(fragFile);

            vk::PipelineShaderStageCreateInfo vertShaderStageInfo = {};
            vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
            vertShaderStageInfo.module = vertShaderModule;
            vertShaderStageInfo.pName = "main";

            vk::PipelineShaderStageCreateInfo fragShaderStageInfo = {};
            fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";

            vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

            auto bindingDescription = fontVertex::getBindingDescription();
            auto attributeDescriptions = fontVertex::getAttributeDescription();

            vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
            vertexInputInfo.vertexBindingDescriptionCount = 1;
            vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
            vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
            vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

            vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {};
            inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            vk::Viewport viewport = {0.0, 0.0, (float) m_swapchainExtent.width, (float) m_swapchainExtent.height, 0.0, 1.0};
            vk::Rect2D scissor = {{0, 0}, m_swapchainExtent};

            vk::PipelineViewportStateCreateInfo viewportState = {};
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;

            vk::PipelineRasterizationStateCreateInfo rasterizer = {};
            rasterizer.depthClampEnable = VK_FALSE;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode = vk::PolygonMode::eFill;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = vk::CullModeFlagBits::eFront;
            rasterizer.frontFace = vk::FrontFace::eCounterClockwise;

            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f;
            rasterizer.depthBiasClamp = 0.0f;

            vk::PipelineMultisampleStateCreateInfo multisampling = {};
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
            multisampling.minSampleShading = 1.0f;
            multisampling.pSampleMask = nullptr;
            multisampling.alphaToCoverageEnable = VK_FALSE;
            multisampling.alphaToOneEnable = VK_FALSE;

            vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
            using CCF = vk::ColorComponentFlagBits;
            colorBlendAttachment.colorWriteMask = CCF::eR | CCF::eG | CCF::eB | CCF::eA;
            colorBlendAttachment.blendEnable = VK_FALSE;

            vk::PipelineColorBlendStateCreateInfo colorBlending = {};
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.logicOp = vk::LogicOp::eCopy;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;

            vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};

            vk::DescriptorSetLayout layout[] = {m_font1DescriptorSetLayout};
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = layout;

            m_fontPipelineLayout = m_device.createPipelineLayout(pipelineLayoutInfo, nullptr);

            vk::GraphicsPipelineCreateInfo pipelineInfo = {};
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;

            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = nullptr;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = nullptr;

            pipelineInfo.layout = m_fontPipelineLayout;
            pipelineInfo.renderPass = m_renderPass;
            pipelineInfo.subpass = 0;

            m_fontPipeline = m_device.createGraphicsPipeline(nullptr, pipelineInfo, nullptr);
            if (!m_fontPipeline)
                throw std::runtime_error("Failed to create pipeline!");

            m_device.destroyShaderModule(fragShaderModule, nullptr);
            m_device.destroyShaderModule(vertShaderModule, nullptr);
        }

        void createObjectPipeline() {
            auto vertFile = compileShaderToSpirv("shaders/objectShader.vert");
            auto fragFile = compileShaderToSpirv("shaders/objectShader.frag");

            auto vertShaderModule = createShaderModule(vertFile);
            auto fragShaderModule = createShaderModule(fragFile);

            vk::PipelineShaderStageCreateInfo vertShaderStageInfo = {};
            vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
            vertShaderStageInfo.module = vertShaderModule;
            vertShaderStageInfo.pName = "main";

            vk::PipelineShaderStageCreateInfo fragShaderStageInfo = {};
            fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";

            vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

            auto bindingDescription = Vertex::getBindingDescription();
            auto attributeDescriptions = Vertex::getAttributeDescription();

            vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
            vertexInputInfo.vertexBindingDescriptionCount = 1;
            vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
            vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
            vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

            vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {};
            inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            vk::Viewport viewport = {0.0, 0.0, (float) m_swapchainExtent.width, (float) m_swapchainExtent.height, 0.0, 1.0};
            vk::Rect2D scissor = {{0, 0}, m_swapchainExtent};

            vk::PipelineViewportStateCreateInfo viewportState = {};
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;

            vk::PipelineRasterizationStateCreateInfo rasterizer = {};
            rasterizer.depthClampEnable = VK_FALSE;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode = vk::PolygonMode::eFill;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = vk::CullModeFlagBits::eBack;
            rasterizer.frontFace = vk::FrontFace::eCounterClockwise;

            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f;
            rasterizer.depthBiasClamp = 0.0f;

            vk::PipelineMultisampleStateCreateInfo multisampling = {};
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
            multisampling.minSampleShading = 1.0f;
            multisampling.pSampleMask = nullptr;
            multisampling.alphaToCoverageEnable = VK_FALSE;
            multisampling.alphaToOneEnable = VK_FALSE;

            vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
            using CCF = vk::ColorComponentFlagBits;
            colorBlendAttachment.colorWriteMask = CCF::eR | CCF::eG | CCF::eB | CCF::eA;
            colorBlendAttachment.blendEnable = VK_FALSE;

            vk::PipelineColorBlendStateCreateInfo colorBlending = {};
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.logicOp = vk::LogicOp::eCopy;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;

            vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};

            vk::DescriptorSetLayout layout[] = {m_rect1DescriptorSetLayout,m_rect2DescriptorSetLayout, m_circleDescriptorSetLayout};
            pipelineLayoutInfo.setLayoutCount = 3;
            pipelineLayoutInfo.pSetLayouts = layout;

            m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutInfo, nullptr);

            vk::GraphicsPipelineCreateInfo pipelineInfo = {};
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;
            
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = nullptr;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = nullptr;

            pipelineInfo.layout = m_pipelineLayout;
            pipelineInfo.renderPass = m_renderPass;
            pipelineInfo.subpass = 0;

            m_graphicsPipeline = m_device.createGraphicsPipeline(nullptr, pipelineInfo, nullptr);
            if (!m_graphicsPipeline)
                throw std::runtime_error("Failed to create pipeline!");

            m_device.destroyShaderModule(fragShaderModule, nullptr);
            m_device.destroyShaderModule(vertShaderModule, nullptr);
        }

        void createFramebuffers() {
            m_swapchainFramebuffers.resize(m_swapchainImageViews.size());

            for (size_t i = 0; i< m_swapchainImageViews.size(); i++) {
                vk::ImageView attachments[] = {
                    m_swapchainImageViews[i]
                };

                vk::FramebufferCreateInfo createInfo = {};
                createInfo.renderPass = m_renderPass;
                createInfo.attachmentCount = 1;
                createInfo.pAttachments = attachments;
                createInfo.width = m_swapchainExtent.width;
                createInfo.height = m_swapchainExtent.height;
                createInfo.layers = 1;

                m_swapchainFramebuffers[i] = m_device.createFramebuffer(createInfo, nullptr);
            }
        }

        void createCommandPool() {
            QueueFamilyIndices queueFamilyIndices = findQueueFamilies(m_physicalDevice);

            vk::CommandPoolCreateInfo poolInfo = {};
            poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

            m_commandPool = m_device.createCommandPool(poolInfo, nullptr);
        }

        vk::CommandBuffer beginSingleTimeCommands() {

            vk::CommandBufferAllocateInfo allocInfo = {};
            allocInfo.level = vk::CommandBufferLevel::ePrimary;
            allocInfo.commandPool = m_commandPool;
            allocInfo.commandBufferCount = 1;

            vk::CommandBuffer commandBuffer;
            m_device.allocateCommandBuffers(&allocInfo, &commandBuffer);

            vk::CommandBufferBeginInfo beginInfo = {};
            beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

            commandBuffer.begin(&beginInfo);

            return commandBuffer;
        }

        void flushSingleTimeCommands(vk::CommandBuffer commandBuffer) {
            commandBuffer.end();

            vk::SubmitInfo submitInfo = {};
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            m_graphicsQueue.submit(1, &submitInfo, nullptr);
            m_graphicsQueue.waitIdle();

            m_device.freeCommandBuffers(m_commandPool, 1, &commandBuffer);
        }

        void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory) {

            vk::ImageCreateInfo imageInfo = {};
            imageInfo.imageType = vk::ImageType::e2D;
            imageInfo.extent.width = width;
            imageInfo.extent.height = height;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = format;
            imageInfo.tiling = tiling;
            imageInfo.initialLayout = vk::ImageLayout::eUndefined;
            imageInfo.usage = usage;
            imageInfo.samples = vk::SampleCountFlagBits::e1;
            imageInfo.sharingMode = vk::SharingMode::eExclusive;
            imageInfo.flags = vk::ImageCreateFlags(0);
            m_device.createImage(&imageInfo, nullptr, &image);

            vk::MemoryRequirements memRequirements;
            m_device.getImageMemoryRequirements(image, &memRequirements);

            vk::MemoryAllocateInfo allocInfo = {};
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
            m_device.allocateMemory(&allocInfo, nullptr, &imageMemory);
            m_device.bindImageMemory(image, imageMemory, 0);
        }

        vk::ImageView createImageView(vk::Image image, vk::Format format) {
            vk::ImageViewCreateInfo viewInfo = {};
            viewInfo.image = image;
            viewInfo.viewType = vk::ImageViewType::e2D;
            viewInfo.format = format;
            viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            vk::ImageView imageView;
            if (m_device.createImageView(&viewInfo, nullptr, &imageView) != vk::Result::eSuccess)
                throw std::runtime_error("Failed to create image view");

            return imageView;
        }

        void createCircleVertexBuffer() {
            using MP_FB = vk::MemoryPropertyFlagBits;
            using BU_FB = vk::BufferUsageFlagBits;

            vk::DeviceSize bufferSize = sizeof(c_circleVertices[0]) * c_circleVertices.size();

            vk::Buffer stagingBuffer;
            vk::DeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, BU_FB::eTransferSrc, MP_FB::eHostVisible | MP_FB::eHostCoherent, stagingBuffer, stagingBufferMemory);

            void* data;
            m_device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags(0), &data);
            memcpy(data, c_circleVertices.data(), (size_t) bufferSize);
            m_device.unmapMemory(stagingBufferMemory);

            createBuffer(bufferSize, BU_FB::eTransferDst | BU_FB::eVertexBuffer, MP_FB::eDeviceLocal, m_circleVertexBuffer, m_circleVertexBufferMemory);

            vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
            copyBuffer(commandBuffer, stagingBuffer, m_circleVertexBuffer, bufferSize);
            flushSingleTimeCommands(commandBuffer);

            m_device.destroy(stagingBuffer);
            m_device.freeMemory(stagingBufferMemory);
        }

        void createRectVertexBuffer() {
            using MP_FB = vk::MemoryPropertyFlagBits;
            using BU_FB = vk::BufferUsageFlagBits;

            vk::DeviceSize bufferSize = sizeof(c_rectVertices[0]) * c_rectVertices.size();

            vk::Buffer stagingBuffer;
            vk::DeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, BU_FB::eTransferSrc, MP_FB::eHostVisible | MP_FB::eHostCoherent, stagingBuffer, stagingBufferMemory);

            void* data;
            m_device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags(0), &data);
            memcpy(data, c_rectVertices.data(), (size_t) bufferSize);
            m_device.unmapMemory(stagingBufferMemory);

            createBuffer(bufferSize, BU_FB::eTransferDst | BU_FB::eVertexBuffer, MP_FB::eDeviceLocal, m_rectVertexBuffer, m_rectVertexBufferMemory);

            vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
            copyBuffer(commandBuffer, stagingBuffer, m_rectVertexBuffer, bufferSize);
            flushSingleTimeCommands(commandBuffer);

            m_device.destroy(stagingBuffer);
            m_device.freeMemory(stagingBufferMemory);
        }

        void createCircleIndexBuffer() {
            using MP_FB = vk::MemoryPropertyFlagBits;
            using BU_FB = vk::BufferUsageFlagBits;

            vk::DeviceSize bufferSize = sizeof(c_circleIndices[0]) * c_circleIndices.size();

            vk::Buffer stagingBuffer;
            vk::DeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, BU_FB::eTransferSrc, MP_FB::eHostVisible | MP_FB::eHostCoherent, stagingBuffer, stagingBufferMemory);

            void* data;
            m_device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags(0), &data);
            memcpy(data, c_circleIndices.data(), (size_t) bufferSize);
            m_device.unmapMemory(stagingBufferMemory);

            createBuffer(bufferSize, BU_FB::eTransferDst | BU_FB::eIndexBuffer, MP_FB::eDeviceLocal, m_circleIndexBuffer, m_circleIndexBufferMemory);

            vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
            copyBuffer(commandBuffer, stagingBuffer, m_circleIndexBuffer, bufferSize);
            flushSingleTimeCommands(commandBuffer);

            m_device.destroy(stagingBuffer);
            m_device.freeMemory(stagingBufferMemory);
        }

        void createRectIndexBuffer() {
            using MP_FB = vk::MemoryPropertyFlagBits;
            using BU_FB = vk::BufferUsageFlagBits;

            vk::DeviceSize bufferSize = sizeof(c_rectIndices[0]) * c_rectIndices.size();

            vk::Buffer stagingBuffer;
            vk::DeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, BU_FB::eTransferSrc, MP_FB::eHostVisible | MP_FB::eHostCoherent, stagingBuffer, stagingBufferMemory);

            void* data;
            m_device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags(0), &data);
            memcpy(data, c_rectIndices.data(), (size_t) bufferSize);
            m_device.unmapMemory(stagingBufferMemory);

            createBuffer(bufferSize, BU_FB::eTransferDst | BU_FB::eIndexBuffer, MP_FB::eDeviceLocal, m_rectIndexBuffer, m_rectIndexBufferMemory);

            vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
            copyBuffer(commandBuffer, stagingBuffer, m_rectIndexBuffer, bufferSize);
            flushSingleTimeCommands(commandBuffer);

            m_device.destroy(stagingBuffer);
            m_device.freeMemory(stagingBufferMemory);
        }

        // Use a separate command pool? (Use CommandPool Transient bit in that case)
        void copyBuffer(vk::CommandBuffer commandBuffer, vk::Buffer f_srcBuffer, vk::Buffer f_dstBuffer, vk::DeviceSize f_size) {
            vk::BufferCopy copyRegion = {};
            copyRegion.size = f_size;
            commandBuffer.copyBuffer(f_srcBuffer, f_dstBuffer, 1, &copyRegion);
        }

        void transitionImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
            vk::ImageMemoryBarrier barrier = {};
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;

            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            barrier.image = image;
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            vk::PipelineStageFlags sourceStage;
            vk::PipelineStageFlags destinationStage;

            if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
                barrier.srcAccessMask = vk::AccessFlags(0);
                barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

                sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
                destinationStage = vk::PipelineStageFlagBits::eTransfer;
            } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
                barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
                barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

                sourceStage = vk::PipelineStageFlagBits::eTransfer;
                destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
            } else {
                throw std::invalid_argument("Unsupported layout transition!");
            }

            commandBuffer.pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlags(0) , 0, nullptr, 0, nullptr, 1, &barrier);
        }

        void copyBufferToImage(vk::CommandBuffer commandBuffer, vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
            vk::BufferImageCopy region = {};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;

            region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;

            region.imageOffset = vk::Offset3D{0, 0, 0};
            region.imageExtent = vk::Extent3D{width, height, 1};

            commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);
        }

        void createBuffer(vk::DeviceSize f_size, vk::BufferUsageFlags f_usage, vk::MemoryPropertyFlags f_properties, vk::Buffer& f_buffer, vk::DeviceMemory& f_bufferMemory) {
            vk::BufferCreateInfo bufferInfo = {};
            bufferInfo.size = f_size;
            bufferInfo.usage = f_usage;
            bufferInfo.sharingMode = vk::SharingMode::eExclusive;

            f_buffer = m_device.createBuffer(bufferInfo);

            vk::MemoryRequirements memRequirements;
            memRequirements = m_device.getBufferMemoryRequirements(f_buffer);

            vk::MemoryAllocateInfo allocInfo = {};
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, f_properties);

            f_bufferMemory = m_device.allocateMemory(allocInfo, nullptr);

            m_device.bindBufferMemory(f_buffer, f_bufferMemory, 0);
        }

        uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
            vk::PhysicalDeviceMemoryProperties memProperties;
            memProperties = m_physicalDevice.getMemoryProperties();
            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
                if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }

            throw std::runtime_error("Failed to find suitable memory type!");
        }

        void createUniformBuffers() {
            vk::DeviceSize mvpBufferSize = sizeof(UniformBufferObject);
            vk::DeviceSize idBufferSize = sizeof(UniformScoreObject);

            m_rect1UniformBuffers.resize(m_swapchainImages.size());
            m_rect1UniformBuffersMemory.resize(m_swapchainImages.size());

            m_rect2UniformBuffers.resize(m_swapchainImages.size());
            m_rect2UniformBuffersMemory.resize(m_swapchainImages.size());

            m_circleUniformBuffers.resize(m_swapchainImages.size());
            m_circleUniformBuffersMemory.resize(m_swapchainImages.size());

            m_text1IDUniformBuffers.resize(m_swapchainImages.size());
            m_text1IDUniformBuffersMemory.resize(m_swapchainImages.size());
            m_text1MVPUniformBuffers.resize(m_swapchainImages.size());
            m_text1MVPUniformBuffersMemory.resize(m_swapchainImages.size());

            m_text2IDUniformBuffers.resize(m_swapchainImages.size());
            m_text2IDUniformBuffersMemory.resize(m_swapchainImages.size());
            m_text2MVPUniformBuffers.resize(m_swapchainImages.size());
            m_text2MVPUniformBuffersMemory.resize(m_swapchainImages.size());

            for (size_t i = 0; i < m_swapchainImages.size(); i++) {
                createBuffer(mvpBufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, m_circleUniformBuffers[i], m_circleUniformBuffersMemory[i]);
                createBuffer(mvpBufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, m_rect1UniformBuffers[i], m_rect1UniformBuffersMemory[i]);
                createBuffer(mvpBufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, m_rect2UniformBuffers[i], m_rect2UniformBuffersMemory[i]);
                createBuffer(mvpBufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, m_text1MVPUniformBuffers[i], m_text1MVPUniformBuffersMemory[i]);
                createBuffer(idBufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, m_text1IDUniformBuffers[i], m_text1IDUniformBuffersMemory[i]);
                createBuffer(mvpBufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, m_text2MVPUniformBuffers[i], m_text2MVPUniformBuffersMemory[i]);
                createBuffer(idBufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, m_text2IDUniformBuffers[i], m_text2IDUniformBuffersMemory[i]);
            }
        }

        void createDescriptorPool() {
            // Object descriptor pool
            {
                std::array<vk::DescriptorPoolSize, 3> poolSizes = {};
                poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
                poolSizes[0].descriptorCount = static_cast<uint32_t>(m_swapchainImages.size());
                poolSizes[1].type = vk::DescriptorType::eUniformBuffer;
                poolSizes[1].descriptorCount = static_cast<uint32_t>(m_swapchainImages.size());
                poolSizes[2].type = vk::DescriptorType::eUniformBuffer;
                poolSizes[2].descriptorCount = static_cast<uint32_t>(m_swapchainImages.size());

                vk::DescriptorPoolCreateInfo poolInfo = {};
                poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
                poolInfo.pPoolSizes = poolSizes.data();
                poolInfo.maxSets = static_cast<uint32_t>(m_swapchainImages.size()) * poolSizes.size();
                poolInfo.flags = vk::DescriptorPoolCreateFlags(0) ;

                m_objectDescriptorPool = m_device.createDescriptorPool(poolInfo);
            }
            // Text descriptor pool
            {
                std::vector<vk::DescriptorPoolSize> poolSizes;

                // Add uniform buffers
                for (int i = 0; i < 4; i++) {
                    poolSizes.push_back(vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(m_swapchainImages.size())});
                }

                // 20 Descriptors
                for (int i = 0; i < 10; i++) {
                    poolSizes.push_back(vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(m_swapchainImages.size())});
                    poolSizes.push_back(vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(m_swapchainImages.size())});
                }

                vk::DescriptorPoolCreateInfo poolInfo = {};
                poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
                poolInfo.pPoolSizes = poolSizes.data();
                poolInfo.maxSets = static_cast<uint32_t>(m_swapchainImages.size()) * poolSizes.size();

                m_textDescriptorPool = m_device.createDescriptorPool(poolInfo);
            }
        }

        void createDescriptorSets() {
            std::vector<vk::DescriptorSetLayout> rect1Layouts(m_swapchainImages.size(), m_rect1DescriptorSetLayout);
            std::vector<vk::DescriptorSetLayout> rect2Layouts(m_swapchainImages.size(), m_rect2DescriptorSetLayout);
            std::vector<vk::DescriptorSetLayout> circleLayouts(m_swapchainImages.size(), m_circleDescriptorSetLayout);
            std::vector<vk::DescriptorSetLayout> font1Layouts(m_swapchainImages.size(), m_font1DescriptorSetLayout);
            std::vector<vk::DescriptorSetLayout> font2Layouts(m_swapchainImages.size(), m_font2DescriptorSetLayout);

            vk::DescriptorSetAllocateInfo rect1AllocInfo = {};
            rect1AllocInfo.descriptorPool = m_objectDescriptorPool;
            rect1AllocInfo.descriptorSetCount = static_cast<uint32_t>(m_swapchainImages.size());
            rect1AllocInfo.pSetLayouts = rect1Layouts.data();

            vk::DescriptorSetAllocateInfo rect2AllocInfo = {};
            rect2AllocInfo.descriptorPool = m_objectDescriptorPool;
            rect2AllocInfo.descriptorSetCount = static_cast<uint32_t>(m_swapchainImages.size());
            rect2AllocInfo.pSetLayouts = rect2Layouts.data();

            vk::DescriptorSetAllocateInfo circleAllocInfo = {};
            circleAllocInfo.descriptorPool = m_objectDescriptorPool;
            circleAllocInfo.descriptorSetCount = static_cast<uint32_t>(m_swapchainImages.size());
            circleAllocInfo.pSetLayouts = circleLayouts.data();

            vk::DescriptorSetAllocateInfo font1AllocInfo = {};
            font1AllocInfo.descriptorPool = m_textDescriptorPool;
            font1AllocInfo.descriptorSetCount = static_cast<uint32_t>(m_swapchainImages.size());
            font1AllocInfo.pSetLayouts = font1Layouts.data();

            vk::DescriptorSetAllocateInfo font2AllocInfo = {};
            font2AllocInfo.descriptorPool = m_textDescriptorPool;
            font2AllocInfo.descriptorSetCount = static_cast<uint32_t>(m_swapchainImages.size());
            font2AllocInfo.pSetLayouts = font2Layouts.data();

            circleDescriptorSets.resize(m_swapchainImages.size());
            if (m_device.allocateDescriptorSets(&circleAllocInfo, circleDescriptorSets.data()) != vk::Result::eSuccess)
                throw std::runtime_error("Failed to allocate Circle descriptor set!");

            rect1DescriptorSets.resize(m_swapchainImages.size());
            if (m_device.allocateDescriptorSets(&rect1AllocInfo, rect1DescriptorSets.data()) != vk::Result::eSuccess)
                throw std::runtime_error("Failed to allocate Rect1 descriptor set!");

            rect2DescriptorSets.resize(m_swapchainImages.size());
            if (m_device.allocateDescriptorSets(&rect2AllocInfo, rect2DescriptorSets.data()) != vk::Result::eSuccess)
                throw std::runtime_error("Failed to allocate Rect2 descriptor set!");

            font1DescriptorSets.resize(m_swapchainImages.size());
            if (m_device.allocateDescriptorSets(&font1AllocInfo, font1DescriptorSets.data()) != vk::Result::eSuccess)
                throw std::runtime_error("Failed to allocate font1 descriptor set!");
            
            font2DescriptorSets.resize(m_swapchainImages.size());
            vk::Result result = m_device.allocateDescriptorSets(&font2AllocInfo, font2DescriptorSets.data());
            if (result != vk::Result::eSuccess) {
                if (result == vk::Result::eErrorOutOfHostMemory)
                    throw std::runtime_error("Out of host memory!");
                else if (result == vk::Result::eErrorOutOfDeviceMemory)
                    throw std::runtime_error("Out of device memory!");
                else if (result == vk::Result::eErrorOutOfPoolMemory)
                    throw std::runtime_error("Out of pool memory!");
                else
                    throw std::runtime_error("Mystery error!");
            }

            for (size_t i = 0; i < m_swapchainImages.size(); i++) {
                vk::DescriptorBufferInfo bufferInfo = {};
                bufferInfo.buffer = m_rect1UniformBuffers[i];
                bufferInfo.offset = 0;
                bufferInfo.range = sizeof(UniformBufferObject);

                std::array<vk::WriteDescriptorSet, 1> descriptorWrites = {};

                descriptorWrites[0].dstSet = rect1DescriptorSets[i];
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &bufferInfo;
                descriptorWrites[0].pImageInfo = nullptr;
                descriptorWrites[0].pTexelBufferView = nullptr;

                m_device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
            }

            for (size_t i = 0; i < m_swapchainImages.size(); i++) {
                vk::DescriptorBufferInfo bufferInfo = {};
                bufferInfo.buffer = m_rect2UniformBuffers[i];
                bufferInfo.offset = 0;
                bufferInfo.range = sizeof(UniformBufferObject);

                std::array<vk::WriteDescriptorSet, 1> descriptorWrites = {};

                descriptorWrites[0].dstSet = rect2DescriptorSets[i];
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &bufferInfo;
                descriptorWrites[0].pImageInfo = nullptr;
                descriptorWrites[0].pTexelBufferView = nullptr;

                m_device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
            }

            for (size_t i = 0; i < m_swapchainImages.size(); i++) {
                vk::DescriptorBufferInfo bufferInfo = {};
                bufferInfo.buffer = m_circleUniformBuffers[i];
                bufferInfo.offset = 0;
                bufferInfo.range = sizeof(UniformBufferObject);

                std::array<vk::WriteDescriptorSet, 1> descriptorWrites = {};

                descriptorWrites[0].dstSet = circleDescriptorSets[i];
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &bufferInfo;
                descriptorWrites[0].pImageInfo = nullptr;
                descriptorWrites[0].pTexelBufferView = nullptr;

                m_device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
            }

            for (size_t i = 0; i < m_swapchainImages.size(); i++) {

                std::array<vk::WriteDescriptorSet, 3> descriptorWrites = {};

                // TODO When replacing with an array of imageInfo, you have to build an array of them in advance
                std::vector<vk::DescriptorImageInfo> imageInfos;
                for(int i = 0; i < 10; i++) {
                    vk::DescriptorImageInfo imageInfo = {};
                    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                    imageInfo.imageView = m_fontImageViews[i];
                    imageInfo.sampler = m_fontSamplers[i];

                    imageInfos.push_back(imageInfo);
                }

                vk::DescriptorBufferInfo mvpBufferInfo = {};
                mvpBufferInfo.buffer = m_text1MVPUniformBuffers[i];
                mvpBufferInfo.offset = 0;
                mvpBufferInfo.range = sizeof(UniformBufferObject);

                vk::DescriptorBufferInfo idBufferInfo = {};
                idBufferInfo.buffer = m_text1IDUniformBuffers[i];
                idBufferInfo.offset = 0;
                idBufferInfo.range = sizeof(UniformScoreObject);

                descriptorWrites[0].dstSet = font1DescriptorSets[i];
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &mvpBufferInfo;
                descriptorWrites[0].pImageInfo = nullptr;
                descriptorWrites[0].pTexelBufferView = nullptr;

                descriptorWrites[1].dstSet = font1DescriptorSets[i];
                descriptorWrites[1].dstBinding = 1;
                descriptorWrites[1].dstArrayElement = 0;
                descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
                descriptorWrites[1].descriptorCount = 10;
                descriptorWrites[1].pBufferInfo = nullptr;
                descriptorWrites[1].pImageInfo = imageInfos.data();
                descriptorWrites[1].pTexelBufferView = nullptr;

                descriptorWrites[2].dstSet = font1DescriptorSets[i];
                descriptorWrites[2].dstBinding = 2;
                descriptorWrites[2].dstArrayElement = 0;
                descriptorWrites[2].descriptorType = vk::DescriptorType::eUniformBuffer;
                descriptorWrites[2].descriptorCount = 1;
                descriptorWrites[2].pBufferInfo = &idBufferInfo;
                descriptorWrites[2].pImageInfo = nullptr;
                descriptorWrites[2].pTexelBufferView = nullptr;

                m_device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
            }

            for (size_t i = 0; i < m_swapchainImages.size(); i++) {

                std::array<vk::WriteDescriptorSet, 3> descriptorWrites = {};

                // TODO When replacing with an array of imageInfo, you have to build an array of them in advance
                std::vector<vk::DescriptorImageInfo> imageInfos;
                for(int i = 0; i < 10; i++) {
                    vk::DescriptorImageInfo imageInfo = {};
                    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                    imageInfo.imageView = m_fontImageViews[i];
                    imageInfo.sampler = m_fontSamplers[i];

                    imageInfos.push_back(imageInfo);
                }

                vk::DescriptorBufferInfo mvpBufferInfo = {};
                mvpBufferInfo.buffer = m_text2MVPUniformBuffers[i];
                mvpBufferInfo.offset = 0;
                mvpBufferInfo.range = sizeof(UniformBufferObject);

                vk::DescriptorBufferInfo idBufferInfo = {};
                idBufferInfo.buffer = m_text2IDUniformBuffers[i];
                idBufferInfo.offset = 0;
                idBufferInfo.range = sizeof(UniformScoreObject);

                descriptorWrites[0].dstSet = font2DescriptorSets[i];
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &mvpBufferInfo;
                descriptorWrites[0].pImageInfo = nullptr;
                descriptorWrites[0].pTexelBufferView = nullptr;

                descriptorWrites[1].dstSet = font2DescriptorSets[i];
                descriptorWrites[1].dstBinding = 1;
                descriptorWrites[1].dstArrayElement = 0;
                descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
                descriptorWrites[1].descriptorCount = 10;
                descriptorWrites[1].pBufferInfo = nullptr;
                descriptorWrites[1].pImageInfo = imageInfos.data();
                descriptorWrites[1].pTexelBufferView = nullptr;

                descriptorWrites[2].dstSet = font2DescriptorSets[i];
                descriptorWrites[2].dstBinding = 2;
                descriptorWrites[2].dstArrayElement = 0;
                descriptorWrites[2].descriptorType = vk::DescriptorType::eUniformBuffer;
                descriptorWrites[2].descriptorCount = 1;
                descriptorWrites[2].pBufferInfo = &idBufferInfo;
                descriptorWrites[2].pImageInfo = nullptr;
                descriptorWrites[2].pTexelBufferView = nullptr;

                m_device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
            }

        }

        void createCommandBuffers() {
            m_commandBuffers.resize(m_swapchainFramebuffers.size());

            vk::CommandBufferAllocateInfo allocInfo = {};
            allocInfo.commandPool = m_commandPool;
            allocInfo.level = vk::CommandBufferLevel::ePrimary;
            allocInfo.commandBufferCount = (uint32_t) m_commandBuffers.size();

            m_commandBuffers = m_device.allocateCommandBuffers(allocInfo);

            for (size_t i = 0; i < m_commandBuffers.size(); i++) {
                vk::CommandBufferBeginInfo beginInfo = {};
                beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;

                m_commandBuffers[i].begin(beginInfo);

                vk::RenderPassBeginInfo renderPassInfo = {};
                renderPassInfo.renderPass = m_renderPass;
                renderPassInfo.framebuffer = m_swapchainFramebuffers[i];

                renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
                renderPassInfo.renderArea.extent = m_swapchainExtent;

                vk::ClearColorValue clearvalueColor;
                clearvalueColor.setFloat32({0.0, 0.0, 0.0, 1.0});
                vk::ClearValue clearColor = clearvalueColor;
                renderPassInfo.clearValueCount = 1;
                renderPassInfo.pClearValues = &clearColor;

                m_commandBuffers[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

                m_commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, m_backgroundPipeline);
                m_commandBuffers[i].draw(3, 1, 0, 0);

                vk::DeviceSize fontOffsets[] = {0};
                m_commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, m_fontPipeline);
                m_commandBuffers[i].bindVertexBuffers(0, 1, &m_fontVertexBuffer, fontOffsets);
                m_commandBuffers[i].bindIndexBuffer(m_fontIndexBuffer, 0, vk::IndexType::eUint16);
                m_commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_fontPipelineLayout, 0, 1, &font1DescriptorSets[i], 0, nullptr);
                m_commandBuffers[i].drawIndexed(static_cast<uint32_t>(c_fontIndices.size()), 1, 0, 0, 0);

                m_commandBuffers[i].bindVertexBuffers(0, 1, &m_fontVertexBuffer, fontOffsets);
                m_commandBuffers[i].bindIndexBuffer(m_fontIndexBuffer, 0, vk::IndexType::eUint16);
                m_commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_fontPipelineLayout, 0, 1, &font2DescriptorSets[i], 0, nullptr);
                m_commandBuffers[i].drawIndexed(static_cast<uint32_t>(c_fontIndices.size()), 1, 0, 0, 0);

                vk::DeviceSize objectOffsets[] = {0};
                m_commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline);
                m_commandBuffers[i].bindVertexBuffers(0, 1, &m_circleVertexBuffer, objectOffsets);
                m_commandBuffers[i].bindIndexBuffer(m_circleIndexBuffer, 0, vk::IndexType::eUint16);
                m_commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, 1, &circleDescriptorSets[i], 0, nullptr);
                m_commandBuffers[i].drawIndexed(static_cast<uint32_t>(c_circleIndices.size()), 1, 0, 0, 0);

                m_commandBuffers[i].bindVertexBuffers(0, 1, &m_rectVertexBuffer, objectOffsets);
                m_commandBuffers[i].bindIndexBuffer(m_rectIndexBuffer, 0, vk::IndexType::eUint16);
                m_commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, 1, &rect1DescriptorSets[i], 0, nullptr);
                m_commandBuffers[i].drawIndexed(static_cast<uint32_t>(c_rectIndices.size()), 1, 0, 0, 0);

                m_commandBuffers[i].bindVertexBuffers(0, 1, &m_rectVertexBuffer, objectOffsets);
                m_commandBuffers[i].bindIndexBuffer(m_rectIndexBuffer, 0, vk::IndexType::eUint16);
                m_commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, 1, &rect2DescriptorSets[i], 0, nullptr);
                m_commandBuffers[i].drawIndexed(static_cast<uint32_t>(c_rectIndices.size()), 1, 0, 0, 0);

                m_commandBuffers[i].endRenderPass();

                m_commandBuffers[i].end();
            }
        }

        void createSyncObjects() {
            m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

            vk::SemaphoreCreateInfo semaphoreInfo = {};

            vk::FenceCreateInfo fenceInfo = {};
            fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                m_imageAvailableSemaphores[i] = m_device.createSemaphore(semaphoreInfo, nullptr);
                m_renderFinishedSemaphores[i] = m_device.createSemaphore(semaphoreInfo, nullptr);
                m_inFlightFences[i] = m_device.createFence(fenceInfo, nullptr);
            }
        }

        vk::ShaderModule createShaderModule(const std::vector<uint32_t>& f_code) {
            vk::ShaderModuleCreateInfo createInfo = {};

            createInfo.codeSize = f_code.size() * sizeof(uint32_t);
            createInfo.pCode = reinterpret_cast<const uint32_t*>(f_code.data());

            return m_device.createShaderModule(createInfo, nullptr);
        }

        vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
                return capabilities.currentExtent;
            } else {
                int v_width, v_height;
                glfwGetFramebufferSize(m_window, &v_width, &v_height);

                vk::Extent2D v_actualExtent = {
                        static_cast<uint32_t>(v_width),
                        static_cast<uint32_t>(v_height)
                };


                v_actualExtent.width = std::clamp(v_actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
                v_actualExtent.height = std::clamp(v_actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

                return v_actualExtent;
            }
        }

        vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> f_availablePresentModes) {
            return vk::PresentModeKHR::eImmediate;
            // TODO Sort this out later, for now 60 fps vsync cap is OK
        }

        vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& f_availableFormats) {
            if (f_availableFormats.size() == 1 && f_availableFormats[0].format == vk::Format::eUndefined) {
                return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
            }

            for (const auto& availableFormat : f_availableFormats) {
                if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                    return availableFormat;
                }
            }

            return f_availableFormats[0];
        }

        bool isDeviceSuitable(vk::PhysicalDevice f_device) {
            if (deviceExtensionsSupported(f_device)) {
                QueueFamilyIndices indices = findQueueFamilies(f_device);
                SwapchainSupportDetails swapchainSupport = querySwapchainSupport(f_device);

                return indices.isComplete() && !swapchainSupport.presentModes.empty(); 
            }

            return false;
        }

        SwapchainSupportDetails querySwapchainSupport(vk::PhysicalDevice f_device) {
            SwapchainSupportDetails details;

            details.capabilities = f_device.getSurfaceCapabilitiesKHR(m_surface);
            details.formats = f_device.getSurfaceFormatsKHR(m_surface);
            details.presentModes = f_device.getSurfacePresentModesKHR(m_surface);

            return details;
        }

        bool deviceExtensionsSupported(vk::PhysicalDevice f_device) {
            std::vector<vk::ExtensionProperties> v_availableExtensions = f_device.enumerateDeviceExtensionProperties();

            std::set<std::string> requiredExtensions(c_deviceExtensions.begin(), c_deviceExtensions.end());

            for (const auto& extension : v_availableExtensions) {
                requiredExtensions.erase(extension.extensionName);
            }

            return requiredExtensions.empty();
        }

        QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice f_device) {
            QueueFamilyIndices indices;

            std::vector<vk::QueueFamilyProperties> queueFamilies = f_device.getQueueFamilyProperties();

            int i = 0;
            for (const auto& queueFamily : queueFamilies) {
                vk::Bool32 presentSupport = false;
                presentSupport = f_device.getSurfaceSupportKHR(i, m_surface);

                if (queueFamily.queueCount > 0) {
                    if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
                        indices.graphicsFamily = i;
                    if (presentSupport)
                        indices.presentFamily = i;
                    if (indices.isComplete())
                        break;
                }

                i++;
            }

            return indices;
        }

        std::vector<const char*> getRequiredExtensions() {
            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
            std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

            if (VALIDATION_LAYERS_ENABLED)
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

            return extensions;
        }


        /*
        // Delta time pseudo-code
        double t = 0.0;
        double dt = 1 / 60.0;
        double currentTime = hires_time_in_seconds();
        while ( !quit )
        {
            double newTime = hires_time_in_seconds();
            double frameTime = newTime - currentTime;
            currentTime = newTime;

            while ( frameTime > 0.0 )
            {
                float deltaTime = min( frameTime, dt );
                integrate( state, t, deltaTime );
                frameTime -= deltaTime;
                t += deltaTime;
            }
            render( state );
        } */

        void mainLoop() {
            glm::normalize(ballVel);
            std::cout << "Ball velocity x: " << ballVel.x << std::endl;
            std::cout << "Ball velocity y: " << ballVel.y << std::endl;
            while (!glfwWindowShouldClose(m_window)) {
                glfwPollEvents();
                integrate(0.016);
                drawFrame();
                std::this_thread::sleep_for(16ms);
            }
        }

        glm::vec2 ballVel = glm::circularRand(1.5f);
        glm::vec2 ballPos = glm::vec2(0.0f, 0.0f);
        uint8_t score1 = 0;
        uint8_t score2 = 0;
        void integrate(double f_deltaTime) {
            float stepSize = 1.6f * f_deltaTime;

            if (keyWPressed)
                paddle1Pos = std::clamp(paddle1Pos + stepSize, -1.5f, 1.5f);

            if (keySPressed)
                paddle1Pos = std::clamp(paddle1Pos - stepSize, -1.5f, 1.5f);

            if (keyUpPressed)
                paddle2Pos = std::clamp(paddle2Pos + stepSize, -1.5f, 1.5f);

            if (keyDownPressed)
                paddle2Pos = std::clamp(paddle2Pos - stepSize, -1.5f, 1.5f);

            // Bounce of top wall
            if (ballPos.y + (ballVel.y * f_deltaTime) >= 1.64f) {
                ballVel.x *= 1.05;
                ballVel.y *= 1.005;
                ballVel.y *= -1;
            }

            // Bounce of bottom wall
            if (ballPos.y + (ballVel.y * f_deltaTime) <= -1.64f) {
                ballVel.x *= 1.05;
                ballVel.y *= 1.005;
                ballVel.y *= -1;
            }

            // Bounce of left paddle
            if (ballPos.x <= -2.75f && (ballPos.y >= paddle1Pos - 0.15f && ballPos.y <= paddle1Pos + 0.15f)) {
                ballVel.x *= 1.1;
                ballVel.y *= 1.01;
                ballVel.x *= -1;
            }

            // Bounce off right paddle
            if (ballPos.x >= 2.75f && (ballPos.y >= paddle2Pos - 0.15f && ballPos.y <= paddle2Pos + 0.15f)) {
                ballVel.x *= 1.1;
                ballVel.y *= 1.01;
                ballVel.x *= -1;
            }

            // Don't let the ball escape
            ballPos.x = std::clamp(ballPos.x + ballVel.x * f_deltaTime, -3.0, 3.0);
            ballPos.y = std::clamp(ballPos.y + ballVel.y * f_deltaTime, -1.65, 1.65);

            if (ballPos.x <= -2.93) {
                score2 = std::clamp(score2 + 1, 0, 9);

                ballPos.x = 0;
                ballPos.y = 0;
                ballVel = glm::circularRand(1.1f * (score1 + score2 + 4)/(4.0));
            }
            else if (ballPos.x >= 2.93) {
                score1 = std::clamp(score1 + 1, 0, 9);

                ballPos.x = 0;
                ballPos.y = 0;
                ballVel = glm::circularRand(1.1f * (score1 + score2 + 4)/4.0);
            }
        }

        void drawFrame() {
            m_device.waitForFences(1, &m_inFlightFences[m_currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

            uint32_t imageIndex;
            vk::Result result = m_device.acquireNextImageKHR(m_swapchain, std::numeric_limits<uint64_t>::max(), m_imageAvailableSemaphores[m_currentFrame], nullptr, &imageIndex);

            if (result == vk::Result::eErrorOutOfDateKHR) {
                recreateSwapchain();
                return;
            } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
                throw std::runtime_error("Failed to acquire swapchain image!");
            }

            vk::Semaphore waitSemaphores[] = {m_imageAvailableSemaphores[m_currentFrame]};
            vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
            vk::Semaphore signalSemaphores[] = {m_renderFinishedSemaphores[m_currentFrame]};

            updateUniformBuffer(imageIndex);

            vk::SubmitInfo submitInfo = {};
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;

            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &m_commandBuffers[imageIndex];

            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            m_device.resetFences(1, &m_inFlightFences[m_currentFrame]);

            if (m_graphicsQueue.submit(1, &submitInfo, m_inFlightFences[m_currentFrame]) != vk::Result::eSuccess)
                throw std::runtime_error("Failed to submit draw command buffer!");

            vk::SwapchainKHR swapchains[] = {m_swapchain};

            vk::PresentInfoKHR presentInfo = {};
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;

            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapchains;
            presentInfo.pImageIndices = &imageIndex;

            result = m_presentQueue.presentKHR(presentInfo);

            if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || m_framebufferResized) {
                m_framebufferResized = false;
                recreateSwapchain();
            } else if (result != vk::Result::eSuccess){
                throw std::runtime_error("Failed to present swapchain image!");
            }

            m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        void updateUniformBuffer(uint32_t f_currentImage) {
            MVPMatrixObject rect1Mvp = {};
            MVPMatrixObject rect2Mvp = {};
            MVPMatrixObject circleMvp = {};
            MVPMatrixObject text1Mvp = {};
            MVPMatrixObject text2Mvp = {};

            UniformBufferObject rect1Ubo = {};
            UniformBufferObject rect2Ubo = {};
            UniformBufferObject circleUbo = {};
            UniformBufferObject text1Ubo = {};
            UniformScoreObject text1Uso = {};
            UniformBufferObject text2Ubo = {};
            UniformScoreObject text2Uso = {};

            glm::mat4 viewMatrix = glm::lookAt(glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 projMatrix = glm::perspective(glm::radians(45.0f), m_swapchainExtent.width / (float) m_swapchainExtent.height, 0.1f, 10.0f);
            projMatrix[1][1] *= -1;

            // Rectangle 1 uniform buffer
            {
                glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(-2.8f, paddle1Pos, 0.0f));
                glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(0.04f, 0.3f, 1.0f));
                glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));
                rect1Mvp.model = translation * rotation * scale;

                rect1Ubo.mvpMatrix = projMatrix * viewMatrix * rect1Mvp.model;

                void* data;
                m_device.mapMemory(m_rect1UniformBuffersMemory[f_currentImage], 0, sizeof(rect1Ubo), vk::MemoryMapFlags(0), &data);
                memcpy(data, &rect1Ubo, sizeof(rect1Ubo));
                m_device.unmapMemory(m_rect1UniformBuffersMemory[f_currentImage]);
            }

            // Rectangle 2 uniform buffer
            {
                glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(2.8f, paddle2Pos, 0.0f));
                glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(0.04f, 0.3f, 1.0f));
                glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));
                rect2Mvp.model = translation * rotation * scale;

                rect2Ubo.mvpMatrix = projMatrix * viewMatrix * rect2Mvp.model;

                void* data;
                m_device.mapMemory(m_rect2UniformBuffersMemory[f_currentImage], 0, sizeof(rect2Ubo), vk::MemoryMapFlags(0), &data);
                memcpy(data, &rect2Ubo, sizeof(rect2Ubo));
                m_device.unmapMemory(m_rect2UniformBuffersMemory[f_currentImage]);
            }

            // Circle uniform buffer
            {
                glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(ballPos.x, ballPos.y, 0.0f));
                glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(0.03f, 0.03f, 1.0f));
                /* glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)); */
                circleMvp.model = translation * scale;

                circleUbo.mvpMatrix = projMatrix * viewMatrix * circleMvp.model;

                void* data;
                m_device.mapMemory(m_circleUniformBuffersMemory[f_currentImage], 0, sizeof(circleUbo), vk::MemoryMapFlags(0), &data);
                memcpy(data, &circleUbo, sizeof(circleUbo));
                m_device.unmapMemory(m_circleUniformBuffersMemory[f_currentImage]);
            }

            // Text/Score 1 uniform
            {
                glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(-1.0,-1.0,0.0));
                glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 1.0f));
                /* glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)); */
                text1Mvp.model = translation * scale;

                projMatrix[1][1] *= -1;
                text1Ubo.mvpMatrix = projMatrix * viewMatrix * text1Mvp.model;

                // MVP uniform
                void* mvpData;
                m_device.mapMemory(m_text1MVPUniformBuffersMemory[f_currentImage], 0, sizeof(text1Ubo), vk::MemoryMapFlags(0), &mvpData);
                memcpy(mvpData, &text1Ubo, sizeof(text1Ubo));
                m_device.unmapMemory(m_text1MVPUniformBuffersMemory[f_currentImage]);

                // ID uniform
                text1Uso.score = score1;
                void* idData;
                m_device.mapMemory(m_text1IDUniformBuffersMemory[f_currentImage], 0, sizeof(text1Uso), vk::MemoryMapFlags(0), &idData);
                memcpy(idData, &text1Uso, sizeof(text1Uso));
                m_device.unmapMemory(m_text1IDUniformBuffersMemory[f_currentImage]);
            }

            // Score 2 uniform
            {
                glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(1.0,-1.0,0.0));
                glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 1.0f));
                /* glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)); */
                text2Mvp.model = translation * scale;

                /* projMatrix[1][1] *= -1; */
                text2Ubo.mvpMatrix = projMatrix * viewMatrix * text2Mvp.model;

                // MVP uniform
                void* mvpData;
                m_device.mapMemory(m_text2MVPUniformBuffersMemory[f_currentImage], 0, sizeof(text2Ubo), vk::MemoryMapFlags(0), &mvpData);
                memcpy(mvpData, &text2Ubo, sizeof(text2Ubo));
                m_device.unmapMemory(m_text2MVPUniformBuffersMemory[f_currentImage]);

                // ID uniform
                text2Uso.score = score2;
                void* idData;
                m_device.mapMemory(m_text2IDUniformBuffersMemory[f_currentImage], 0, sizeof(text2Uso), vk::MemoryMapFlags(0), &idData);
                memcpy(idData, &text2Uso, sizeof(text2Uso));
                m_device.unmapMemory(m_text2IDUniformBuffersMemory[f_currentImage]);
            }
        }

        void recreateSwapchain() {
            int width = 0, height = 0;
            while (width == 0 || height == 0) {
                glfwGetFramebufferSize(m_window, &width, &height);
                glfwWaitEvents();
            }

            vkDeviceWaitIdle(m_device);

            cleanupSwapchain();

            createSwapchain();
            createImageViews();
            createRenderPass();
            createObjectPipeline();
            createBackgroundPipeline();
            createTextPipeline();
            createFramebuffers();
            createCommandBuffers();
        }

        void cleanupSwapchain() {

            for (auto framebuffer : m_swapchainFramebuffers) {
                m_device.destroy(framebuffer);
            }

            m_device.free(m_commandPool, m_commandBuffers);

            m_device.destroy(m_graphicsPipeline);
            m_device.destroy(m_pipelineLayout);

            m_device.destroy(m_backgroundPipeline);
            m_device.destroy(m_backgroundPipelineLayout);

            m_device.destroy(m_fontPipeline);
            m_device.destroy(m_fontPipelineLayout);

            m_device.destroy(m_renderPass);

            for (auto imageView : m_swapchainImageViews) {
                m_device.destroy(imageView);
            }

            m_device.destroy(m_swapchain);
        }

        void cleanup() {
            vkDeviceWaitIdle(m_device);

            cleanupSwapchain();

            m_device.destroy(m_objectDescriptorPool);
            m_device.destroy(m_textDescriptorPool);

            m_device.destroy(m_rect1DescriptorSetLayout);
            m_device.destroy(m_rect2DescriptorSetLayout);
            m_device.destroy(m_circleDescriptorSetLayout);
            m_device.destroy(m_font1DescriptorSetLayout);
            m_device.destroy(m_font2DescriptorSetLayout);

            for (size_t i = 0; i < m_swapchainImages.size(); i++) {
                m_device.destroy(m_rect1UniformBuffers[i]);
                m_device.free(m_rect1UniformBuffersMemory[i]);
                m_device.destroy(m_rect2UniformBuffers[i]);
                m_device.free(m_rect2UniformBuffersMemory[i]);

                m_device.destroy(m_circleUniformBuffers[i]);
                m_device.free(m_circleUniformBuffersMemory[i]);

                m_device.destroy(m_text1IDUniformBuffers[i]);
                m_device.free(m_text1IDUniformBuffersMemory[i]);
                m_device.destroy(m_text1MVPUniformBuffers[i]);
                m_device.free(m_text1MVPUniformBuffersMemory[i]);

                m_device.destroy(m_text2IDUniformBuffers[i]);
                m_device.free(m_text2IDUniformBuffersMemory[i]);
                m_device.destroy(m_text2MVPUniformBuffers[i]);
                m_device.free(m_text2MVPUniformBuffersMemory[i]);
            }

            for (int i = 0; i < 10; i++) {
                m_device.destroy(m_fontImages[i]);
                m_device.destroy(m_fontImageViews[i]);
                m_device.destroy(m_fontSamplers[i]);
                m_device.free(m_fontImagesMemory[i]);
            }

            m_device.destroy(m_circleIndexBuffer);
            m_device.free(m_circleIndexBufferMemory);
            m_device.destroy(m_circleVertexBuffer);
            m_device.free(m_circleVertexBufferMemory);

            m_device.destroy(m_rectIndexBuffer);
            m_device.free(m_rectIndexBufferMemory);
            m_device.destroy(m_rectVertexBuffer);
            m_device.free(m_rectVertexBufferMemory);

            m_device.destroy(m_fontIndexBuffer);
            m_device.free(m_fontIndexBufferMemory);
            m_device.destroy(m_fontVertexBuffer);
            m_device.free(m_fontVertexBufferMemory);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                m_device.destroy(m_renderFinishedSemaphores[i]);
                m_device.destroy(m_imageAvailableSemaphores[i]);
                m_device.destroy(m_inFlightFences[i]);
            }

            m_device.destroy(m_commandPool);

            m_device.destroy();

            if (VALIDATION_LAYERS_ENABLED)
                m_instance.destroyDebugUtilsMessengerEXT(m_messenger, nullptr, vk::DispatchLoaderDynamic{m_instance});

            m_instance.destroySurfaceKHR(m_surface);
            m_instance.destroy();

            glfwDestroyWindow(m_window);
            glfwTerminate();
        }

        void initWindow() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

            m_window = glfwCreateWindow(1280, 720, "vkPong", nullptr, nullptr);

            glfwSetWindowUserPointer(m_window, this);
            glfwSetKeyCallback(m_window, key_callback);
            glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
        }

        // TODO Figure out and utilize compiler options
        std::vector<uint32_t> compileShaderToSpirv(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::in);

            if (!file.is_open())
                throw std::runtime_error("Failed to open file: " + filename + "!");

            size_t fileSize = (size_t) file.tellg();
            std::vector<char> buffer(fileSize);

            file.seekg(0);
            file.read(buffer.data(), fileSize);
            file.close();

            std::vector<uint32_t> spirv;
            shaderc::Compiler compiler;
            if (!compiler.IsValid())
                throw std::runtime_error("Shaderc compiler isn't valid!");

            auto shaderType = determineShaderType(filename);
            if (shaderType == shaderc_glsl_infer_from_source)
                throw std::runtime_error("Failed to determine shader type!");

            std::string str(buffer.begin(), buffer.end());
            auto result = compiler.CompileGlslToSpv(str, shaderType, filename.c_str());
            if (result.GetCompilationStatus() != shaderc_compilation_status_success)
            {
                std::cerr << "Failed shaderc(" << result.GetCompilationStatus() << "): " << result.GetErrorMessage() << std::endl;
                throw std::runtime_error("Failed to compile shader: " + filename);
            }
            spirv.assign(result.cbegin(), result.cend());

            return spirv;
        }

        shaderc_shader_kind determineShaderType(const std::string& filename) {

            std::string reverseString = filename;
            std::reverse(reverseString.begin(), reverseString.end());
            std::string ending = filename.substr(filename.length() - reverseString.find("."));

            if (ending == "vert")
                return shaderc_vertex_shader;
            else if (ending == "frag")
                return shaderc_fragment_shader;
            else if (ending == "tesc")
                return shaderc_tess_control_shader;
            else if (ending == "tese")
                return shaderc_tess_evaluation_shader;
            else if (ending == "geom")
                return shaderc_geometry_shader;
            else if (ending == "comp")
                return shaderc_compute_shader;

            return shaderc_shader_kind::shaderc_glsl_infer_from_source;
        }
};

int main() {
    srand(time(0));

    application v_app;
    try {
        v_app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
