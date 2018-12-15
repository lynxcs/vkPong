// TODO Built-in shader compilation
// TODO Optimize vector usage
// TODO Use SPIRV-Reflect to fill out pipeline layout

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <optional>
#include <set>
#include <algorithm>

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

        vk::PipelineLayout m_pipelineLayout;
        vk::Pipeline m_graphicsPipeline;

        vk::CommandPool m_commandPool;
        std::vector<vk::CommandBuffer> m_commandBuffers;

        std::vector<vk::Semaphore> m_imageAvailableSemaphores;
        std::vector<vk::Semaphore> m_renderFinishedSemaphores;
        std::vector<vk::Fence> m_inFlightFences;

        int m_currentFrame = 0;

        bool m_framebufferResized = false;

        struct QueueFamilyIndices {
            std::optional<uint32_t> graphicsFamily;
            std::optional<uint32_t> presentFamily;

            bool isComplete() {
                return graphicsFamily.has_value() && presentFamily.has_value();
            }
        };

        struct SwapchainSupportDetails {
            vk::SurfaceCapabilitiesKHR capabilities;
            std::vector<vk::SurfaceFormatKHR> formats;
            std::vector<vk::PresentModeKHR> presentModes;
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
            createGraphicsPipeline();
            createFramebuffers();
            createCommandPool();
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
            for (size_t i = 0; i < m_swapchainImages.size(); i++) {
                vk::ImageViewCreateInfo createInfo = {};
                createInfo.image = m_swapchainImages[i];

                createInfo.viewType = vk::ImageViewType::e2D;
                createInfo.format = m_swapchainImageFormat;

                // TODO Check if all components need to be set individualy
                createInfo.components = vk::ComponentSwizzle::eIdentity;

                createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
                createInfo.subresourceRange.baseMipLevel = 0;
                createInfo.subresourceRange.levelCount = 1;
                createInfo.subresourceRange.baseArrayLayer = 0;
                createInfo.subresourceRange.layerCount = 1;

                m_swapchainImageViews[i] = m_device.createImageView(createInfo, nullptr);
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

        void createGraphicsPipeline() {
            auto vertShaderCode = readFile("shaders/vert.spv");
            auto fragShaderCode = readFile("shaders/frag.spv");

            vk::ShaderModule vertShaderModule;
            vk::ShaderModule fragShaderModule;

            vertShaderModule = createShaderModule(vertShaderCode);
            fragShaderModule = createShaderModule(fragShaderCode);

            vk::PipelineShaderStageCreateInfo vertShaderStageInfo = {};
            vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
            vertShaderStageInfo.module = vertShaderModule;
            vertShaderStageInfo.pName = "main";

            vk::PipelineShaderStageCreateInfo fragShaderStageInfo = {};
            fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";

            vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

            vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
            vertexInputInfo.vertexBindingDescriptionCount = 0;
            vertexInputInfo.pVertexBindingDescriptions = nullptr;
            vertexInputInfo.vertexAttributeDescriptionCount = 0;
            vertexInputInfo.pVertexAttributeDescriptions = nullptr;

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
            rasterizer.frontFace = vk::FrontFace::eClockwise;

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
                m_commandBuffers[i].draw(3, 1, 0, 0);
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

        vk::ShaderModule createShaderModule(const std::vector<char>& f_code) {
            vk::ShaderModuleCreateInfo createInfo = {};
            createInfo.codeSize = f_code.size();
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
