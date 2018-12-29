// TODO Built-in shader compilation
// TODO Use SPIRV-Reflect to fill out pipeline layout
// TODO Use dedicated transfer queue

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <shaderc/shaderc.hpp>

#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <optional>
#include <set>
#include <algorithm>
#include <chrono>

#ifdef NDEBUG
const bool VALIDATION_LAYERS_ENABLED = false;
#else
const bool VALIDATION_LAYERS_ENABLED = true;
#endif

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> c_validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> c_deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

static void key_callback(GLFWwindow* f_window, int f_key, int f_scancode, int f_action, int f_mods) {
    if (f_key == GLFW_KEY_ESCAPE && f_action == GLFW_PRESS) {
        glfwSetWindowShouldClose(f_window, GLFW_TRUE);
    }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + filename + "!");

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
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

        vk::DescriptorSetLayout m_descriptorSetLayout;
        vk::DescriptorPool m_descriptorPool;
        std::vector<vk::DescriptorSet> descriptorSets;

        vk::PipelineLayout m_pipelineLayout;
        vk::Pipeline m_graphicsPipeline;

        vk::CommandPool m_commandPool;
        std::vector<vk::CommandBuffer> m_commandBuffers;

        std::vector<vk::Semaphore> m_imageAvailableSemaphores;
        std::vector<vk::Semaphore> m_renderFinishedSemaphores;
        std::vector<vk::Fence> m_inFlightFences;

        vk::Buffer m_vertexBuffer;
        vk::DeviceMemory m_vertexBufferMemory;

        vk::Buffer m_indexBuffer;
        vk::DeviceMemory m_indexBufferMemory;

        std::vector<vk::Buffer> m_uniformBuffers;
        std::vector<vk::DeviceMemory> m_uniformBuffersMemory;

        vk::Image m_textureImage;
        vk::ImageView m_textureImageView;
        vk::DeviceMemory m_textureImageMemory;
        vk::Sampler m_textureSampler;

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
            glm::vec2 pos;
            glm::vec3 color;
            glm::vec2 texCoord;

            static vk::VertexInputBindingDescription getBindingDescription() {
                vk::VertexInputBindingDescription bindingDescription = {};
                bindingDescription.binding = 0;
                bindingDescription.stride = sizeof(Vertex);
                bindingDescription.inputRate = vk::VertexInputRate::eVertex;

                return bindingDescription;
            }

            static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescription() {
                std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions = {};
                attributeDescriptions[0].binding = 0;
                attributeDescriptions[0].location = 0;
                attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
                attributeDescriptions[0].offset = offsetof(Vertex, pos);

                attributeDescriptions[1].binding = 0;
                attributeDescriptions[1].location = 1;
                attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
                attributeDescriptions[1].offset = offsetof(Vertex, color);

                attributeDescriptions[2].binding = 0;
                attributeDescriptions[2].location = 2;
                attributeDescriptions[2].format = vk::Format::eR32G32Sfloat;
                attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

                return attributeDescriptions;
            }
        };

        const std::vector<Vertex> c_vertices = {
            {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
            {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
        };

        // TODO Calculate these automatically
        const std::vector<uint16_t> c_indices = {
            0, 1, 2, 2, 3, 0
        };

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
            createGraphicsPipeline();
            createFramebuffers();
            createCommandPool();
            createTextureImage();
            createTextureImageView();
            createTextureSampler();
            createVertexBuffer();
            createIndexBuffer();
            createUniformBuffers();
            createDescriptorPool();
            createDescriptorSets();
            createCommandBuffers();
            createSyncObjects();
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
            vk::DescriptorSetLayoutBinding uboLayoutBinding = {};
            uboLayoutBinding.binding = 0;
            uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
            uboLayoutBinding.descriptorCount = 1;

            uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
            uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

            // TODO: Maybe also use one in vertex for heightmap?
            vk::DescriptorSetLayoutBinding samplerLayoutBinding = {};
            samplerLayoutBinding.binding = 1;
            samplerLayoutBinding.descriptorCount = 1;
            samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
            samplerLayoutBinding.pImmutableSamplers = nullptr;
            samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

            std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
            vk::DescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();

            m_descriptorSetLayout = m_device.createDescriptorSetLayout(layoutInfo);
        }

        void createGraphicsPipeline() {
            auto vertFile = compileShaderToSpirv("shaders/shader.vert");
            auto fragFile = compileShaderToSpirv("shaders/shader.frag");

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
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;

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

        void createTextureImage() {
            int texWidth, texHeight, texChannels;
            stbi_uc* pixels = stbi_load("textures/texture.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
            vk::DeviceSize imageSize = texWidth * texHeight * 4;

            if (!pixels)
                throw std::runtime_error("Failed to load texture image!");

            vk::Buffer stagingBuffer;
            vk::DeviceMemory stagingBufferMemory;

            using BU = vk::BufferUsageFlagBits;
            using MP = vk::MemoryPropertyFlagBits;
            createBuffer(imageSize, BU::eTransferSrc, MP::eHostVisible | MP::eHostCoherent, stagingBuffer, stagingBufferMemory);

            void * data;
            m_device.mapMemory(stagingBufferMemory, 0, imageSize, vk::MemoryMapFlags(0), &data);
            memcpy(data, pixels, static_cast<size_t>(imageSize));
            m_device.unmapMemory(stagingBufferMemory);

            stbi_image_free(pixels);

            createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, m_textureImage, m_textureImageMemory);

            vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
            transitionImageLayout(commandBuffer, m_textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            copyBufferToImage(commandBuffer, stagingBuffer, m_textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
            transitionImageLayout(commandBuffer, m_textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
            flushSingleTimeCommands(commandBuffer);

            m_device.destroy(stagingBuffer);
            m_device.free(stagingBufferMemory);
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

        void createTextureImageView() {
            m_textureImageView = createImageView(m_textureImage, vk::Format::eR8G8B8A8Unorm);
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
            m_device.createImageView(&viewInfo, nullptr, &imageView);

            return imageView;
        }

        void createTextureSampler() {
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

            m_device.createSampler(&samplerInfo, nullptr, &m_textureSampler);
        }

        void createVertexBuffer() {
            using MP_FB = vk::MemoryPropertyFlagBits;
            using BU_FB = vk::BufferUsageFlagBits;

            vk::DeviceSize bufferSize = sizeof(c_vertices[0]) * c_vertices.size();

            vk::Buffer stagingBuffer;
            vk::DeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, BU_FB::eTransferSrc, MP_FB::eHostVisible | MP_FB::eHostCoherent, stagingBuffer, stagingBufferMemory);

            void* data;
            m_device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags(0), &data);
            memcpy(data, c_vertices.data(), (size_t) bufferSize);
            m_device.unmapMemory(stagingBufferMemory);

            createBuffer(bufferSize, BU_FB::eTransferDst | BU_FB::eVertexBuffer, MP_FB::eDeviceLocal, m_vertexBuffer, m_vertexBufferMemory);

            vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
            copyBuffer(commandBuffer, stagingBuffer, m_vertexBuffer, bufferSize);
            flushSingleTimeCommands(commandBuffer);

            m_device.destroy(stagingBuffer);
            m_device.freeMemory(stagingBufferMemory);
        }

        void createIndexBuffer() {
            using MP_FB = vk::MemoryPropertyFlagBits;
            using BU_FB = vk::BufferUsageFlagBits;

            vk::DeviceSize bufferSize = sizeof(c_indices[0]) * c_indices.size();

            vk::Buffer stagingBuffer;
            vk::DeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, BU_FB::eTransferSrc, MP_FB::eHostVisible | MP_FB::eHostCoherent, stagingBuffer, stagingBufferMemory);

            void* data;
            m_device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags(0), &data);
            memcpy(data, c_indices.data(), (size_t) bufferSize);
            m_device.unmapMemory(stagingBufferMemory);

            createBuffer(bufferSize, BU_FB::eTransferDst | BU_FB::eIndexBuffer, MP_FB::eDeviceLocal, m_indexBuffer, m_indexBufferMemory);

            vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
            copyBuffer(commandBuffer, stagingBuffer, m_indexBuffer, bufferSize);
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
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

            m_uniformBuffers.resize(m_swapchainImages.size());
            m_uniformBuffersMemory.resize(m_swapchainImages.size());

            for (size_t i = 0; i < m_swapchainImages.size(); i++) {
                createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, m_uniformBuffers[i], m_uniformBuffersMemory[i]);
            }
        }

        void createDescriptorPool() {
            std::array<vk::DescriptorPoolSize, 2> poolSizes = {};
            poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
            poolSizes[0].descriptorCount = static_cast<uint32_t>(m_swapchainImages.size());
            poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
            poolSizes[1].descriptorCount = static_cast<uint32_t>(m_swapchainImages.size());
            
            vk::DescriptorPoolCreateInfo poolInfo = {};
            poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
            poolInfo.pPoolSizes = poolSizes.data();
            poolInfo.maxSets = static_cast<uint32_t>(m_swapchainImages.size());
            poolInfo.flags = vk::DescriptorPoolCreateFlags(0) ;

            m_descriptorPool = m_device.createDescriptorPool(poolInfo);
        }

        void createDescriptorSets() {
            std::vector<vk::DescriptorSetLayout> layouts(m_swapchainImages.size(), m_descriptorSetLayout);

            vk::DescriptorSetAllocateInfo allocInfo = {};
            allocInfo.descriptorPool = m_descriptorPool;
            allocInfo.descriptorSetCount = static_cast<uint32_t>(m_swapchainImages.size());
            allocInfo.pSetLayouts = layouts.data();

            descriptorSets.resize(m_swapchainImages.size());
            m_device.allocateDescriptorSets(&allocInfo, descriptorSets.data());

            for (size_t i = 0; i < m_swapchainImages.size(); i++) {
                vk::DescriptorBufferInfo bufferInfo = {};
                bufferInfo.buffer = m_uniformBuffers[i];
                bufferInfo.offset = 0;
                bufferInfo.range = sizeof(UniformBufferObject);

                vk::DescriptorImageInfo imageInfo = {};
                imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                imageInfo.imageView = m_textureImageView;
                imageInfo.sampler = m_textureSampler;

                std::array<vk::WriteDescriptorSet, 2> descriptorWrites = {};

                descriptorWrites[0].dstSet = descriptorSets[i];
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &bufferInfo;
                descriptorWrites[0].pImageInfo = nullptr;
                descriptorWrites[0].pTexelBufferView = nullptr;

                descriptorWrites[1].dstSet = descriptorSets[i];
                descriptorWrites[1].dstBinding = 1;
                descriptorWrites[1].dstArrayElement = 0;
                descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
                descriptorWrites[1].descriptorCount = 1;
                descriptorWrites[1].pBufferInfo = nullptr;
                descriptorWrites[1].pImageInfo = &imageInfo;
                descriptorWrites[1].pTexelBufferView = nullptr;

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
                m_commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline);

                vk::Buffer vertexBuffers[] = {m_vertexBuffer};
                vk::DeviceSize offsets[] = {0};
                m_commandBuffers[i].bindVertexBuffers(0, 1, vertexBuffers, offsets);
                m_commandBuffers[i].bindIndexBuffer(m_indexBuffer, 0, vk::IndexType::eUint16);
                m_commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

                m_commandBuffers[i].drawIndexed(static_cast<uint32_t>(c_indices.size()), 1, 0, 0, 0);
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
            return vk::PresentModeKHR::eFifo;
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

        void mainLoop() {
            while (!glfwWindowShouldClose(m_window)) {
                glfwPollEvents();
                drawFrame();
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
            static auto startTime = std::chrono::high_resolution_clock::now();

            auto currentTime = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

            UniformBufferObject ubo = {};
            ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.proj = glm::perspective(glm::radians(45.0f), m_swapchainExtent.width / (float) m_swapchainExtent.height, 0.1f, 10.0f);
            ubo.proj[1][1] *= -1;

            void* data;
            m_device.mapMemory(m_uniformBuffersMemory[f_currentImage], 0, sizeof(ubo), vk::MemoryMapFlags(0), &data);
            memcpy(data, &ubo, sizeof(ubo));
            m_device.unmapMemory(m_uniformBuffersMemory[f_currentImage]);
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
            createGraphicsPipeline();
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
            m_device.destroy(m_renderPass);

            for (auto imageView : m_swapchainImageViews) {
                m_device.destroy(imageView);
            }

            m_device.destroy(m_swapchain);
        }

        void cleanup() {
            vkDeviceWaitIdle(m_device);

            cleanupSwapchain();

            m_device.destroy(m_textureSampler);
            m_device.destroy(m_textureImageView);

            m_device.destroy(m_textureImage);
            m_device.free(m_textureImageMemory);

            m_device.destroy(m_descriptorPool);

            m_device.destroy(m_descriptorSetLayout);
            for (size_t i = 0; i < m_swapchainImages.size(); i++) {
                m_device.destroy(m_uniformBuffers[i]);
                m_device.free(m_uniformBuffersMemory[i]);
            }

            m_device.destroy(m_indexBuffer);
            m_device.free(m_indexBufferMemory);

            m_device.destroy(m_vertexBuffer);
            m_device.free(m_vertexBufferMemory);

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

            m_window = glfwCreateWindow(800, 600, "Vulkan", nullptr, nullptr);

            glfwSetWindowUserPointer(m_window, this);
            glfwSetKeyCallback(m_window, key_callback);
            glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
        }

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
    application v_app;

    try {
        v_app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
